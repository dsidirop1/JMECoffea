import pandas as pd
import numpy as np

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
    check_hist_values(hist1, hist2)
    return {key:(hist1[key]*hist2[key]) for key in hist1.keys()}

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

def read_data(name, flavor, tag):
    df_csv = pd.read_csv('out_txt/EtaBinsvsPtBins'+name+'_'+flavor+tag+'.csv').set_index('etaBins')
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
        combine_samples = [flavor for flavor in all_samples if 'untagged' not in flavor ]
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

