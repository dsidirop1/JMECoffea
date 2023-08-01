import numpy as np
from coffea import util
from helpers import hist_div, sum_subhist, add_flavors
from JetEtaBins import JetEtaBins


def get_output(data_tag):
    ''' Runs util.load with the correct file name with the tag `data_tag`
    '''
    tag_full = '_L5_'+data_tag
    outname = 'out/CoffeaJERCOutputs'+tag_full+'.coffea'
    output = util.load(outname)
    return output

# output_orig = output
def sum_output(output, data_tag, xsec_dict):
    ''' If the file with histograms `output` contains a dictionary over many datasets (e.g. different pt ranges),
    sum them up proportionally to the cross sections in `file_dict` 
    Output: summed up hisograms `output`
    '''
    
    keys = output.keys()
    Nev = {key: output[key]['cutflow']['all_events'].value for key in keys}
    # response_sums = {key:sum(dictionary_pattern(output[key], "ptresponse_").values()).sum().value for key in output.keys()}
    # scale_factors = {key:1 for key in output.keys()} #hist_div(xsec_dict, Nev)
    scale_factors = hist_div(xsec_dict, Nev)
    all_histo_keys = output[next(iter(output.keys()))].keys()
    hists_merged = {histo_key:sum_subhist(output, histo_key, scale_factors) for histo_key in all_histo_keys }  
    return hists_merged
    
#     if "_QCD-MG" in data_tag:
#         Nev = {key: output[key]['cutflow']['all_events'].value for key in output.keys()}
# #         response_sums = {key:sum(dictionary_pattern(output[key], "ptresponse_").values()).sum().value for key in output.keys()}
#         scale_factors = hist_div(file_dict, Nev)
#         HT50key = list(dictionary_pattern(file_dict, "HT50").keys())[0]
#         scale_factors[HT50key] = 0
#         all_histo_keys = output[next(iter(output.keys()))].keys()
#         result = {histo_key:sum_subhist(output, histo_key, scale_factors) for histo_key in all_histo_keys }
#         output = result
#     elif len(output.keys())==1:
#         output = output[list(output)[0]]
#     return output

def combine_flavors(output, sumeta=True, include_unmatched=True):
    flavors = (['g', 'c', 'b', 'ud', 's', 'unmatched', 'all_unmatched'] if include_unmatched==True
               else ['g', 'c', 'b', 'ud', 's', 'all'])
    hists = {}
    for flav in flavors:
        combined = add_flavors(output, flavor=flav, combine_antiflavour=True )[0]
        if sumeta:
            combined = combined[:,sum,sum]
        else:
            combined = combined[:,sum,:]
        flav2 = flav if not flav=='all_unmatched' else 'all'
        hists[flav2] = combined
    return hists

from dataclasses import dataclass, field

def convert_to_np_array(var):
    if not type(var) is np.ndarray:
        var = np.array(var)
    if len(var.shape)==0:
        var = np.array([var])
    return var

ptmin_global = 30
ptmax_global = 500

@dataclass
class FlavorFractions():
    E_frac_splines: dict
    binning: str = "HCalPart"
        
    flavors: str = field(init=False, repr=True)
    def __post_init__(self):
        if not isinstance(self.E_frac_splines, dict):
            raise TypeError(f"The argument in E_frac_splines has to be a dictionary over the flavors. The given type is {type(self.E_frac_splines)}")
        self.flavors = np.array([key for key in self.E_frac_splines.keys()])
  
    def evaluate(self, flav, etavals, ptvals):
        ''' Evaluate the flavor fractions for flavor `flav` and at `etavals` and `ptvals`
        '''
        splines = self.E_frac_splines[flav]
        
        etavals = np.abs(etavals)
        etavals = convert_to_np_array(etavals)
        ptvals = convert_to_np_array(ptvals)
                    
        ptvals = np.clip(ptvals, ptmin_global, ptmax_global)
        logptvals = np.log10(ptvals)
        eta_idxs = JetEtaBins(self.binning, absolute=True).get_bin_idx(etavals) #JERC_Constants.GetEtaBinIdx(etavals, self.binning)        
        #usually quicker to loop over the same eta indices at the same time than through all the incides
        outs = np.ones(shape=(etavals.size), dtype=np.float32)
        for i in np.unique(eta_idxs): 
            mask = np.where(eta_idxs == i)
            splineii   = splines[i]
            logptvalsii   = logptvals[mask]
            logptvalsii  = np.clip(logptvalsii, np.min(splineii.x), np.max(splineii.x) )
            
            outs[mask] = splineii(logptvalsii)
        return outs
    
    def get_limits(self, etavals, flav=None):
        ''' Get the pt limits of the splines with flavor `flav` at `etavals`
        '''
        if flav==None:
            lims = np.array([self.get_limits_flav(etavals, flav) for flav in self.flavors])
            ptmins = lims[:,0,:]
            ptmaxs = lims[:,1,:]
            ptmins = np.max(ptmins, axis=0)
            ptmaxs = np.min(ptmaxs, axis=0)
        else:
            ptmins, ptmaxs = self.get_limits_flav(etavals, flav)
        return ptmins, ptmaxs
            
    
    def get_limits_flav(self, etavals, flav):
        ''' Get the pt limits of the splines with flavor `flav` at `etavals`
        '''
        
        splines = self.E_frac_splines[flav]
        
        etavals = np.abs(etavals)
        etavals = convert_to_np_array(etavals)
        eta_idxs = JetEtaBins(self.binning, absolute=True).get_bin_idx(etavals)
        spline_eta = splines[eta_idxs]
        ptmins = np.power(10, [np.min(splineii.x) for splineii in spline_eta])
        ptmaxs = np.power(10, [np.max(splineii.x) for splineii in spline_eta])
        return ptmins, ptmaxs
    
    def get_x(self):
        return self.E_frac_splines[self.flavors[0]][0].x


from scipy.interpolate import CubicSpline
def get_spline(yval, pt_bins):
    valid_fit_val = ~(np.isnan(yval) | np.isinf(yval) | (yval==0))
    ptbins_c_plot = pt_bins.centres[valid_fit_val]
    yval = yval[valid_fit_val]

    spline_func = CubicSpline(np.log10(ptbins_c_plot), yval, bc_type='natural', extrapolate=False )
    return spline_func

def get_ratio(a, b, divide=True):
    '''To obtain the JEC uncertainty instead of dividing Herwig by Python, the samples are subtracted'''
    if divide:
        return(a/b)
    else:
        return(a-b)

from helpers import read_data

def read_data2(name, sample, flav, eta_binning_str):
    data = read_data(name, flav, "_L5"+sample+eta_binning_str)
    data[data==0] = np.nan
    return data

def read_corrections(sample, flav, eta_binning_str):
    data = read_data("Median", flav, "_L5"+sample+eta_binning_str)
    data[data==0] = np.nan
    return 1/data

def create_data_HerPy_differences(sampHer, sampPy, pt_idxs, eta_idxs, inverse=False, divideHerPy=False, eta_binning_str='Summer20Flavor'):
    sampHer = '_'+sampHer
    sampPy = '_'+sampPy
    if inverse==True:
        a = read_corrections(sampHer, 'all')
        b = read_corrections(sampPy, 'all')
        delta_a = read_data2('MedianStd', sampHer, 'all', eta_binning_str)
        delta_b = read_data2('MedianStd', sampPy, 'all', eta_binning_str)
        corr_all = get_ratio(a, b, divideHerPy)[pt_idxs,eta_idxs]
        if divideHerPy:
            corrstd_all = np.sqrt((delta_a*b)**2+(delta_b*b**2/a)**2)[pt_idxs,eta_idxs]
        else:
            corrstd_all = np.sqrt((delta_a*a**2)**2+(delta_b*b**2)**2)[pt_idxs,eta_idxs]
    else:
        a = read_data2('Median', sampHer, 'all', eta_binning_str)
        b = read_data2('Median', sampPy, 'all', eta_binning_str)
        delta_a = read_data2('MedianStd', sampHer, 'all', eta_binning_str)
        delta_b = read_data2('MedianStd', sampPy, 'all', eta_binning_str)
        corr_all = get_ratio(a, b, divideHerPy)[pt_idxs,eta_idxs]
        if divideHerPy:
            corrstd_all = np.sqrt((delta_a/b)**2+(a/b**2*delta_b)**2)[pt_idxs,eta_idxs]
        else:
            corrstd_all = np.sqrt(delta_a**2+delta_b**2)[pt_idxs,eta_idxs]
    return [corr_all, corrstd_all]

# def get_evaluator_pre(evaluator, correction_txt, sample='Her', flavor='b' , inverse=False, use_corrections='J', eta_binning_str='Summer20Flavor'):
#     '''
#     Get the required funciton from evaluator with the required sample and flavor.     
#     inverse==True gets a correction, inverse==False gets a response.
#     '''
#     samp = '_Her' if sample=='Her' else ''         
#     eva = evaluator[f'{correction_txt}{samp}{eta_binning_str}_{flavor}{use_corrections}']
#     if inverse==True:
#         return eva
#     else:
#         return lambda a,b: 1/eva(a,b)
    
# def get_evaluator_limits(evaluator, correction_txt, samp, flav, use_corrections,  etavals):
#     '''Get the intersect of the pt ranges of all the evaluators for the given etavals
#     samp = 'Her' or 'Py'
#     flav = flavor
#     use_corrections = 'J' for dijet or 'T' for ttbar
#     '''
#     corr_etabins = get_evaluator_pre(evaluator, correction_txt, samp, flav, inverse=True, use_corrections=use_corrections)._bins['JetEta'] 
#     corr_bin_idx = np.searchsorted(corr_etabins, etavals, side='right')-1
#     corr_bin_idx[corr_bin_idx>=len(corr_etabins)] = len(corr_etabins)-1
    
#     ptmins = list(get_evaluator_pre(samp, flav, inverse=True, use_corrections=use_corrections)._eval_clamp_mins.values())[0]
#     ptmaxs = list(get_evaluator_pre(samp, flav, inverse=True, use_corrections=use_corrections)._eval_clamp_maxs.values())[0]
#     ptmins = np.array(ptmins)[corr_bin_idx, 0]
#     ptmaxs = np.array(ptmaxs)[corr_bin_idx, 0]
#     return ptmins, ptmaxs
    
# def get_evaluator_limits_all_flav(evaluator, correction_txt, use_corrections,  etavals, flavors):
#     ''' Get the intersect of the pt ranges of all the evaluators for the given etavals for all flavors and for Herwig and Pythia evaluators. 
#     '''
#     lims = np.array([get_evaluator_limits(evaluator, correction_txt, 'Her', flav, use_corrections, etavals) for flav in flavors])
#     ptmins = lims[:,0,:]
#     ptmaxs = lims[:,1,:]
#     ptmins = np.max(ptmins, axis=0)
#     ptmaxs = np.min(ptmaxs, axis=0)
#     lims = np.array([get_evaluator_limits(evaluator, correction_txt, 'Py', flav, use_corrections, etavals) for flav in flavors])
#     ptmins = np.max(np.vstack([ptmins, lims[:,0,:]]),axis=0)
#     ptmaxs = np.min(np.vstack([ptmaxs, lims[:,1,:]]),axis=0)
#     return ptmins, ptmaxs


# def resum_to_mix_one_flav(etavals, ptvals, Efracspline, samp, flav, etavals_frac, ptvals_frac):

#     response = get_evaluator(samp, flav)(etavals, ptvals)
#     return response*Efracspline.evaluate(flav, etavals_frac, ptvals_frac)

# def resum_to_mix_from_ratio_one_flav(etavals, ptvals, Efracspline, Efracspline2, flav, etavals_frac, ptvals_frac):
# #     if etavals_frac==None or ptvals_frac==None:
# #         etavals_frac = etavals
# #         ptvals_frac = ptvals 
#     response = Her_Py_ratio_fit_res.evaluate(flav, etavals, ptvals )
#     if Efracspline2==None:
#         return response*Efracspline.evaluate(flav, etavals_frac, ptvals_frac)
#     else:
#         return response*(Efracspline.evaluate(flav,etavals_frac, ptvals_frac)+
#                                Efracspline2.evaluate(flav,etavals_frac, ptvals_frac)/2)

# def resum_to_mix(Efracspline, samp, etavals, ptvals, etavals_frac=None, ptvals_frac=None):
#     ''' Sum up the corrections from the fit according to the flavor fractions in `Efracspline`.
#     samp = 'Her' or 'Py' for Herwig corrections or for Pythia corrections
#     '''
#     spline_sum = sum(
#         resum_to_mix_one_flav(etavals, ptvals, Efracspline, samp, flav, etavals_frac, ptvals_frac  ) for flav in flavors
#     )
#     return spline_sum

# def resum_ratio_to_mix(etavals, ptvals, Efracspline, Efracspline2=None, divideHerPy=False, start_from_ratios=True,
#                        etavals_frac=None, ptvals_frac=None):
#     ''' Compute Eq. (26) in Sec. 7 of arXiv:1607.03663 at etavals and ptvals.
#     Use flavor fractions splines given in Efracspline, Efracspline2.
#     `divideHerPy` == True: calculate the Herwig/Pythia ratio; False: calculate the Herwig/Pythia difference
#     `start_from_ratios` == True: resum the Herwig/Pythia ratios to the average of the flavor content;
#                            False: do as in Eq. (26)
#     '''
    
#     ptvals, etavals = convert_to_np_array(ptvals), convert_to_np_array(etavals)
#     pt_min, pt_max = Efracspline.get_limits(etavals)
#     if not Efracspline2==None:
#         pt_min2, pt_max2 = Efracspline2.get_limits(etavals)
#     else:
#         pt_min2, pt_max2 = pt_min, pt_max
        
#     pt_min3, pt_max3 = get_evaluator_limits_all_flav(evaluator, correction_txt, use_corrections, etavals)
#     pt_min_tot = np.max([pt_min, pt_min2, pt_min3, [ptmin_global]*len(pt_min)],axis=0)
#     pt_max_tot = np.min([pt_max, pt_max2, pt_max3, [ptmax_global]*len(pt_min)],axis=0)
        
#     ptvals = np.clip(ptvals, pt_min_tot, pt_max_tot)
# #     assert False
# #     print(ptvals)

#     if etavals_frac==None or ptvals_frac==None:
#         etavals_frac = etavals
#         ptvals_frac = ptvals
#     if start_from_ratios:
#         return Her_Py_ratio_fit_res.resum_to_mix(etavals, ptvals, Efracspline, Efracspline2, etavals_frac, ptvals_frac )
#     else:
#         return get_ratio(resum_to_mix(Efracspline, 'Her', etavals, ptvals, etavals_frac, ptvals_frac),
#                          resum_to_mix(Efracspline2, 'Py', etavals, ptvals, etavals_frac, ptvals_frac),
#                          divideHerPy)
    
# one_spline = get_spline(np.array([1]*pt_bins.nbins), pt_bins)
# zero_spline = get_spline(np.array([1e-15]*pt_bins.nbins), pt_bins) # does not work if using 0

# def get_additional_uncertainty_curves(etavals, ptvals, etavals0, dijetat_eta0, start_from_ratios=True):
#     '''Obtain all the other curves neccessary for the plots'''
#     result = {}
#     result["g20q80"] = resum_ratio_to_mix(etavals, ptvals,
#                                           qfrac_spline_dict['DY-MG-Her'],
#                                           qfrac_spline_dict['DY-MG-Py'],
#                                           divideHerPy,
#                                           start_from_ratios,
#                                           0, 200
#                                          )
#     result["g20q80_fixed"] = resum_ratio_to_mix(0, 200,
#                                           qfrac_spline_dict['DY-MG-Her'],
#                                           qfrac_spline_dict['DY-MG-Py'],
#                                           divideHerPy,
#                                           start_from_ratios
#                                          )
    
#     for flav in flavors:
#         fractions100_tmp = {flavii: np.array([zero_spline]*jeteta_bins.nbins) for flavii in flavors if flav not in flavii}
#         fractions100_tmp[flav] = np.array([one_spline]*jeteta_bins.nbins)
#         fractions100 = FlavorFractions(fractions100_tmp, eta_binning)
        
#         result[flav+'100'] = resum_ratio_to_mix(etavals, ptvals,
#                               fractions100,
#                               fractions100,
#                               divideHerPy,
#                               start_from_ratios
#                              )

#     Rdijet0 = resum_ratio_to_mix(etavals0, ptvals,
#                                   qfrac_spline_dict['QCD-MG-Her'],
#                                   qfrac_spline_dict['QCD-MG-Py'],
#                                   divideHerPy,
#                                   start_from_ratios
#                                 )
    
#     result["Rref"] = result["g20q80_fixed"] + (dijetat_eta0 - Rdijet0) #HerPy_differences['QCD'][0]
    
#     return result