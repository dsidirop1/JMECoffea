import pandas as pd
import numpy as np

def rebin_hist(h, axis_name, edges):
    '''Stolen from Kenneth Long
    https://gist.github.com/kdlong/d697ee691c696724fc656186c25f8814
    '''
    if type(edges) == int:
        return h[{axis_name : hist.rebin(edges)}]

    ax = h.axes[axis_name]

    if len(ax.edges)==len(edges) and (ax.edges == edges).all():
        return h

    ax_idx = [a.name for a in h.axes].index(axis_name)
    if not all([np.isclose(x, ax.edges).any() for x in edges]):
        raise ValueError(f"Cannot rebin histogram due to incompatible edges for axis '{ax.name}'\n"
                            f"Edges of histogram are {ax.edges}, requested rebinning to {edges}")
        
    # If you rebin to a subset of initial range, keep the overflow and underflow
    overflow = ax.traits.overflow or (edges[-1] < ax.edges[-1] and not np.isclose(edges[-1], ax.edges[-1]))
    underflow = ax.traits.underflow or (edges[0] > ax.edges[0] and not np.isclose(edges[0], ax.edges[0]))
    flow = overflow or underflow
    new_ax = hist.axis.Variable(edges, name=ax.name, overflow=overflow, underflow=underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax
    
    hnew = hist.Hist(*axes, name=h.name, storage=h.storage_type())

    # Offset from bin edge to avoid numeric issues
    offset = 0.5*np.min(ax.edges[1:]-ax.edges[:-1])
    edges_eval = edges+offset
    edge_idx = ax.index(edges_eval)
    # Avoid going outside the range, reduceat will add the last index anyway
    if edge_idx[-1] == ax.size+ax.traits.overflow:
        edge_idx = edge_idx[:-1]

    if underflow:
        # Only if the original axis had an underflow should you offset
        if ax.traits.underflow:
            edge_idx += 1
        edge_idx = np.insert(edge_idx, 0, 0)

    # Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value. Add tolerance of 1/2 min bin width to avoid numeric issues
    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx, 
            axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    if hnew.storage_type() == hist.storage.Weight():
        hnew.variances(flow=flow)[...] = np.add.reduceat(h.variances(flow=flow), edge_idx, 
                axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    return hnew


def mirror_eta_to_plus(h):
    ''' Mirror negative eta bins in histogram to positive eta bins
    '''
    axis_name = 'jeteta'
    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)

    overflow = ax.traits.overflow
    underflow = ax.traits.underflow
    flow = overflow or underflow
    new_eta_edges = h.axes[ax_idx].edges[::-1]*(-1)
    #     if (new_eta_edges>0).any() and (new_eta_edges<0).any():
    #         Maybe a bug?
    new_ax = hist.axis.Variable(new_eta_edges, name=ax.name, overflow=overflow, underflow=underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax

    hnew = hist.Hist(*axes, name=h.name, storage=h.storage_type())

    hnew.values(flow=flow)[...]=np.flip(h.values(flow=flow),axis=ax_idx)
    if hnew.storage_type() == hist.storage.Weight():
        hnew.variances(flow=flow)[...] = np.flip(h.variances(flow=flow),axis=ax_idx)
    return hnew

def sum_neg_pos_eta(hist):
    eta_edges = [ax.edges for ax in hist.axes if 'jeteta' in ax.name ][0]
    # ed = response_hist.axes.edges[0]
    edPl = eta_edges[eta_edges>=0]
    edMi = eta_edges[eta_edges<=0]

    hist_Pl = rebin_hist(hist, 'jeteta', edPl)
    hist_Mi = rebin_hist(hist, 'jeteta', edMi)

    hist_MiPl = mirror_eta_to_plus(hist_Mi)
    return hist_MiPl+hist_Pl


################ Dictionary sum helper functions #######################
# Why isn't anything like this implemented in Python itself?

def dictionary_pattern(dictionary, pattern):
    "skim the histogram with keys mathing the pattern"
    return {key:dictionary[key] for key in dictionary.keys() if pattern in key}

def check_hist_values(hist1, hist2):
    if not (hist1.keys()==hist2.keys()):
        raise ValueError(f"Keys of the two histograms are not the same. The keys are {hist1.keys()} and {hist2.keys()}")

def hist_add(hist1, hist2):
    check_hist_values(hist1, hist2)
    return {key:(hist1[key]+hist2[key]) for key in hist1.keys()}


def hist_mult(hist1, hist2):
    if type(hist2)==dict:
        check_hist_values(hist1, hist2)
        return {key:(hist1[key]*hist2[key]) for key in hist1.keys()}
    else:
        return {key:(hist1[key]*hist2) for key in hist1.keys()}

def hist_div(hist1, hist2):
    check_hist_values(hist1, hist2)
    return {key:(hist1[key]/hist2[key]) for key in hist1.keys()}

def sum_subhist(output, histo_key, scales):
    ''' Merge the datasets in dictionary `output` for `histo_key` and scale each dataset with `scales`
    '''
    new_hist = 0    
    for sample_key in scales.keys():
        new_hist += output[sample_key][histo_key]*scales[sample_key]
    return new_hist


def save_data(data, name, flavor, tag, ptbins, etabins):
    data_dict = {str(ptBin):data[i] for i, ptBin in enumerate(ptbins[:-1])}
    data_dict['etaBins'] = np.array([str(etaBin) for etaBin in etabins[:-1]])

    df = pd.DataFrame(data=data_dict)
    df = df.set_index('etaBins')
    df.to_csv('out_txt/EtaBinsvsPtBins'+name+'_'+flavor+tag+'.csv')

def read_data(name, flavor, tag, rel_path=''):
    df_csv = pd.read_csv(rel_path+'out_txt/EtaBinsvsPtBins'+name+'_'+flavor+tag+'.csv').set_index('etaBins')
    return df_csv.to_numpy().transpose()

def xsecstr2float(str_input):
    str_split = str_input.split('*')
    try:
        if len(str_split)==1:
            xsec = float(str_split[0])
        elif len(str_split)==2:
            xsec = float(str_split[0])*float(str_split[1])
        else:
            raise ValueError
    except:
        raise ValueError("Check your input files. Cross-secton not correctly defined."+
                         f" It has to be either a number or a number times a factor. Given: {str_input}")
    return xsec

def find_ttbar_xsec(key):
    '''Used for combining different ttbar channels
    '''
    semilepxsec = 0.108*3*0.337*2*2
    dilepxsec = 0.108*3*0.108*3
    hadxsec = 0.337*2*0.337*2

    if "TTToSemi" in key:
        xsec = semilepxsec
    elif "TTToDilep" in key:
        xsec = dilepxsec
    elif "TTToHad" in key:
        xsec = hadxsec
    else:
        xsec = 1
    return xsec


def get_median(xvals, yvals, bin_edges, Neff):
    ''' Calculate median and median error (assuming Gaussian distribution).
    This is the binned median, not the real data median
    Extrapolation withing bins is performed.
    '''
    yvals_cumsum = np.cumsum(yvals)
    N = np.sum(yvals)

    # once adding weights, Neff appears to be ~1/4 - 1/3 of N when not using weights,
    # so changing limits to match the both cases
    if np.abs(np.sum(yvals)-Neff)/Neff<1e-5:
        N_min_limit=200
    else:
        N_min_limit=50

    if Neff>N_min_limit:
        med_bin = np.nonzero(yvals_cumsum>N/2)[0][0]
        median = bin_edges[med_bin] + (N/2 - yvals_cumsum[med_bin-1])/yvals[med_bin]*(bin_edges[med_bin+1]
                                                                                 - bin_edges[med_bin])
    else:
        median = 0

    hist_mean = np.sum(xvals*yvals)/sum(yvals) 
    hist_rms = np.sqrt(np.sum(yvals*((hist_mean-xvals)**2))/sum(yvals))
    medianstd = 1.253 * hist_rms/np.sqrt(Neff)
    
    return median, medianstd


barable_samples = ['b', 'c', 's', 'u', 'd']

composite_sample_dict = {
    'q': ['u', 'd', 's'],
    'ud': ['u', 'd'],
    'qbar': ['ubar', 'dbar', 'sbar'],
    'udbar': ['ubar', 'dbar'],
}

def add_flavors(output, flavor='all', combine_antiflavour=True ):
    ''' Sum the response and pt histograms in `output` according to the schema in `composite_sample_dict`.
    Return the summed up histograms `response_hist`, `recopt_hist`.
    '''
    all_samples = [key[11:] for key in output.keys() if 'ptresponse_' in key]
    if 'bar' in flavor and combine_antiflavour:
        raise ValueError(f"combine_antiflavour is set to True but the sample {flavor} contains bar. This might lead to inconsistencies.")
    
    
    ############## Find the correct histograms from output to add ################
    if flavor=='all':
        combine_samples = [flavor for flavor in all_samples if 'unmatched' not in flavor ]
    elif flavor=='all_unmatched':
        combine_samples = [flavor for flavor in all_samples ]
    else:
        try:
            combine_samples = composite_sample_dict[flavor]
        except KeyError:
            combine_samples = [flavor]
        if combine_antiflavour:
            combine_samples_bar = [flavor+'bar' for flavor in combine_samples if flavor in barable_samples]
            combine_samples = combine_samples_bar + combine_samples

    all_responses = {flavor:output['ptresponse_'+flavor] for flavor in combine_samples}
    response_hist = sum(all_responses.values())
    all_reco_pts = {flavor:output['reco_pt_sumwx_'+flavor] for flavor in combine_samples}
    recopt_hist = sum(all_reco_pts.values())

    return response_hist, recopt_hist

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

import hist
def slice_histogram(response_hist, recopt_hist, etabins, etaidx, pt_lo, pt_hi):
    '''To do: 
    - make a switch from either combining eta>0 and eta<0 bins or not. So far, combing them. 
    '''
    jeteta_length = (len(etabins)-1)//2
    etaPl_lo = etabins[etaidx+jeteta_length]
    etaPl_hi = etabins[etaidx+1+jeteta_length]
    etaMi_lo = etabins[jeteta_length-etaidx-1]
    etaMi_hi = etabins[jeteta_length-etaidx]
    eta_string = '_eta'+str(etaPl_lo)+'to'+str(etaPl_hi)
    eta_string = eta_string.replace('.','')

        
    sliceMi = {'jeteta': slice(hist.loc(etaMi_lo),hist.loc(etaMi_hi),sum),
                'pt_gen': slice(hist.loc(pt_lo),hist.loc(pt_hi),sum)}
    slicePl = {'jeteta': slice(hist.loc(etaPl_lo),hist.loc(etaPl_hi),sum),
                'pt_gen': slice(hist.loc(pt_lo),hist.loc(pt_hi),sum)}

    histoMi = response_hist[sliceMi]
    histoPl = response_hist[slicePl]
    histo = (histoMi+histoPl)
    
    histoptMi = recopt_hist[sliceMi]
    histoptPl = recopt_hist[slicePl]
    histopt = (histoptMi+histoptPl)

    return histo, histopt, eta_string

