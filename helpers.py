import pandas as pd
import numpy as np

from pathlib import Path
from scipy.optimize import curve_fit

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

def sum_subhist(histo, histo_key, scales):
    ''' Merge (stitch) the datasets in the dictionary `histo` for `histo_key` and weighting each dataset by the scale factor defined in `scales`.
    `histo`: a dictionary of dictionaries (corresponding to each dataset) of the histograms (correspinding to each histogram).
    `histo_key`: merge the datasets for the given histogram in histo
    `scales`: a dictionary of the scale factors for each dataset.
    returns: a histogram for `histo_key` which all the datasets merged
    '''
    new_hist = 0    
    for sample_key in scales.keys():
        new_hist += histo[sample_key][histo_key]*scales[sample_key]
    return new_hist

# def scale_subhist(histo, histo_key, scales):
#     ''' Scale the datasets in the dictionary `histo` for `histo_key` and scale each dataset with `scales`
#     '''
#     new_hist = {sample_key:{} for sample_key in scales.keys()}    
#     for sample_key in scales.keys():
#         new_hist[sample_key][histo_key] = histo[sample_key][histo_key]*scales[sample_key]
#     return new_hist

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

def save_data(data, name, flavor, tag, ptbins_centres, etabins, path='out_txt'):
    data_dict = {str(ptBin):data[i] for i, ptBin in enumerate(ptbins_centres)}
    data_dict['etaBins'] = np.array([str(etaBin) for etaBin in etabins[:-1]])

    df = pd.DataFrame(data=data_dict)
    df = df.set_index('etaBins')
    df.to_csv(path+'/EtaBinsvsPtBins'+name+'_'+flavor+tag+'.csv')

def read_data(name, flavor, tag, path='out_txt'):
    df_csv = pd.read_csv(path+'/EtaBinsvsPtBins'+name+'_'+flavor+tag+'.csv').set_index('etaBins')
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


def get_median(histo, Neff):
    ''' Calculate median and median error (assuming Gaussian distribution).
    This is the binned median, not the real data median
    Extrapolation withing bins is performed.
    '''
    xvals = histo.axes[0].centers
    bin_edges = histo.axes[0].edges
    yvals = histo.values()
    yvals_cumsum = np.cumsum(yvals)
    N = np.sum(yvals)

    # once adding weights, Neff appears to be ~1/4 - 1/3 of N when not using weights,
    # so changing limits to match the both cases
    # if np.abs(np.sum(yvals)-Neff)/Neff<1e-5:
    #     N_min_limit=200
    # else:
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

def fit_response(histo, Neff, Nfit=3, sigma_fit_window=1.5):
    ''' fit response distribution with `Nfit` consecutive gaussian fits
    Perform the second and further fits around the mean+/-<sigma_fit_window>*std of the previous fit.

    '''
    if_failed = False   #save if the fit failed or converged
    
    xvals = histo.axes[0].centers
    yvals = histo.values()
    variances = histo.variances()
    
    # once adding weights, Neff appears to be ~1/4 - 1/3 of N when not using weights,
    # so changing limits to match the both cases
    # if (np.sum(yvals)-Neff)/Neff<1e-5:
    #     N_min_limit=50
    # else:
    N_min_limit=50
    
    nonzero_bins = np.sum(yvals>0)
    if nonzero_bins<2 or Neff<N_min_limit:
        p=[0,0,0]
        chi2 = np.nan
        cov = np.array([[np.nan]*3]*3)
        Ndof = 0
        if_failed = True
        xfit_l, xfit_h = [0, len(xvals)-1]
    else:
        try:
            p, cov = curve_fit(gauss, xvals, yvals, p0=[10,1,1])
#             print("p vals 0 = ", p)
            ######## Second Gaussian ########
            for i in range(Nfit-1):
                xfit_l, xfit_h = np.searchsorted(xvals,
                                                 [p[1]-np.abs(p[2])*sigma_fit_window,
                                                  p[1]+np.abs(p[2])*sigma_fit_window])
                
                # if there are only 3pnts, the uncertainty is infty
                # (or if too small, then the fit doesn't become good),
                # so increase the range
                rangeidx = 0
                while len(range(xfit_l,xfit_h))<6 and rangeidx<3:
                    xfit_l = xfit_l-1
                    xfit_h = xfit_h+1
                    rangeidx+=1
                if xfit_l<0:
                    xfit_h-=xfit_l
                    xfit_l = 0
                xvals2 = xvals[xfit_l:xfit_h]
                yvals2 = yvals[xfit_l:xfit_h]
                p, cov = curve_fit(gauss, xvals2, yvals2, p0=p)
#                 print(f"p vals {i+1} = ", p)
                 ######## End second Gaussian ########

            ygaus = gauss(xvals, *p)
            chi2 = sum(((yvals-ygaus)**2/(variances+1E-20))[xfit_l:xfit_h] )
            Ndof = len(xvals2)-3
        except(RuntimeError):   #When fit failed
            p=[0,0,0]
            chi2 = np.nan
            cov = np.array([[np.nan]*3]*3)
            Ndof = 0
            if_failed = True
            xfit_l, xfit_h = [0, len(xvals)-1]
            
    return [p, cov, chi2, Ndof, if_failed, [xfit_l, xfit_h]]


barable_flavors = ['b', 'c', 's', 'u', 'd', 'ud', 'q']

composite_flavor_dict = {
    'q': ['u', 'd', 's'],
    'ud': ['u', 'd'],
    'qbar': ['ubar', 'dbar', 'sbar'],
    'udbar': ['ubar', 'dbar'],
}

def get_flavor_antiflavor_list(flavors):
    flavors_new = []
    for flav in flavors:
        if flav in barable_flavors:
            flavors_new.append(flav)
            flavors_new.append(flav+'bar')
    return flavors_new

def add_flavors(output, flavor='all', combine_antiflavour=True ):
    ''' Sum the response and pt histograms in `output` according to the schema in `composite_flavor_dict`.
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
            combine_samples = composite_flavor_dict[flavor]
        except KeyError:
            combine_samples = [flavor]
        if combine_antiflavour:
            combine_samples_bar = [flavor+'bar' for flavor in combine_samples if flavor in barable_flavors]
            combine_samples = combine_samples_bar + combine_samples

    all_responses = {flavor:output['ptresponse_'+flavor] for flavor in combine_samples}
    response_hist = sum(all_responses.values())
    all_reco_pts = {flavor:output['reco_pt_sumwx_'+flavor] for flavor in combine_samples}
    recopt_hist = sum(all_reco_pts.values())

    return response_hist, recopt_hist

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def get_xsecs_filelist_from_file(file_path, data_tag, test_run=False):
    with open(file_path) as f:
        lines = f.readlines()
    lines_split = [line.split() for line in lines]
    if test_run:
        lines_split = lines_split[:3]  
    xsec_dict = {data_tag+'_'+lineii[1]: xsecstr2float(lineii[2]) for lineii in lines_split }
    file_dict = {data_tag+'_'+lineii[1]: lineii[0] for lineii in lines_split }
    return xsec_dict, file_dict


def get_xsec_dict(data_tag, dataset_dictionary):
    ''' Load a text file with cross sections and file names as a dictionary `xsec_dict`.
    '''
    ### if the 'data_tag' in the root contains any of the tags in `dataset_dictionary`, select this tag,
    ### e.g.,'QCD-Py_weights' contains 'QCD-Py', so select xsec from 'QCD-Py'.
    keys = np.array(list(dataset_dictionary.keys()))
    matching_keys =  keys[np.where([ key in data_tag for key in keys])[0]]
    if len(matching_keys)>1:
        raise ValueError(f"More than one key from the dataset dictionary matches the given data_tag = {data_tag}")
    elif len(matching_keys)==1:
        matching_key = matching_keys[0]
        dataset, xsec, label = dataset_dictionary[matching_key]
        if (dataset is None) and (xsec is not None):
            xsec_dict, file_dict = get_xsecs_filelist_from_file(xsec, matching_key)
        else:
            xsec_dict = {matching_key: 1}
        legend_label = label
    else:
        xsec_dict = {data_tag: 1}
        legend_label = data_tag
        
    return xsec_dict, legend_label

def find_result_file_index(ResultFile):
    ResultFileName = Path.cwd() / ResultFile
    if ResultFileName.exists():
        Lines = ResultFileName.read_text().split('\n')
        for ii in range(len(Lines)):
            if 'Run_index_' == Lines[-ii-1][:10]:
                RunIndex = int(Lines[-ii-1][10:]) + 1
                break
            if ii==len(Lines):
                print("Couldn't read the index number from"+ResultFile)
    else:
        RunIndex = 1
        
    return RunIndex
