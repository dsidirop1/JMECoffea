from common_binning import JERC_Constants
from JetEtaBins import JetEtaBins, PtBins
from dataclasses import dataclass, field
from uncertainty_helpers import convert_to_np_array, get_ratio, get_spline, FlavorFractions
import numpy as np

def poly4(x, *p):
    c0, c1, c2, c3, c4 = p
    xs = np.log10(x)
    res = c0+c1*xs+c2*xs**2+c3*xs**3+c4*xs**4
    return res

def poly4lims(x, xmin, xmax, *p):
    xcp = x.copy()
    lo_pos = xcp<xmin
    hi_pos = xcp>xmax
    xcp[lo_pos] = xmin
    xcp[hi_pos] = xmax
    return poly4(xcp, *p)

@dataclass
class RatioPlotFitRes():
    ''' A class to easily store the fit results of the ratio plots and to easily evaluate them
    on a similar footing as the corrections read from the .txt files by the coffea evaluator
    
    '''
    binning: str = "Win14"
    fit_coefs: dict = None #field(init=False, repr=True)
#     xmins: dict = field(init=False, repr=True)
#     xmaxs: dict = field(init=False, repr=True)
    flavors: str = np.array([]) # field(init=False, repr=True)
    ptmin_global: float = 30
    ptmax_global: float = 500
    _jetetabins: JetEtaBins = field(init=False, repr=False)
        
    def __post_init__(self):
        binning = self.binning
        if not binning in JERC_Constants.StrToBinsDict().keys():
            raise TypeError(f"The provided eta binning, {binning} not defined in the `common_binning` file. The available binnings are {JERC_Constants.StrToBinsDict().keys()}")
        self._jetetabins = JetEtaBins(binning, absolute=True)
        if not isinstance(self.fit_coefs, (dict, type(None))):
            raise TypeError(f"The argument in fit_coefs has to be a dictionary over the flavors."
                            +f"The given type is {type(self.fit_coefs)}")
#         self.flavors = np.array([])
        if self.fit_coefs==None:
            self.fit_coefs = {}
        else:
            self.flavors = np.array([key for key in self.fit_coefs.keys()])
            
  
    def evaluate(self, flav, etavals, ptvals):
        ''' Evaluate the flavor fractions for flavor `flav` and at `etavals` and `ptvals`
        '''
        
        etaidxs = self._jetetabins.get_bin_idx(etavals)
        ptvals, etaidxs = convert_to_np_array(ptvals), convert_to_np_array(etaidxs) 
        fit_coefs = self.fit_coefs[flav][etaidxs]
        return np.array(
            [poly4lims(np.array(ptval), xfitmin, xfitmax, *p) 
                   for ptval, (p, xfitmin, xfitmax) in zip(ptvals, fit_coefs)]
        )
    
    def add_flavor(self, flav, coefs):
        self.flavors = np.append(self.flavors, flav)
        self.fit_coefs[flav] = coefs

    def resum_to_mix_one_flav(self, etavals, ptvals, Efracspline, Efracspline2, flav, etavals_frac, ptvals_frac):
        response = self.evaluate(flav, etavals, ptvals )
        if Efracspline2==None:
            return response*Efracspline.evaluate(flav, etavals_frac, ptvals_frac)
        else:
            return response*(Efracspline.evaluate(flav,etavals_frac, ptvals_frac)+
                                Efracspline2.evaluate(flav,etavals_frac, ptvals_frac)/2)

    def resum_to_mix(self, etavals, ptvals, Efracspline, Efracspline2, etavals_frac, ptvals_frac ):

        spline_sum = sum(
                self.resum_to_mix_one_flav(etavals, ptvals, Efracspline, Efracspline2, flav, etavals_frac, ptvals_frac )   for flav in self.flavors
            )
        return spline_sum.T

    def resum_ratio_to_mix(self, etavals, ptvals, Efracspline_Her, Efracspline_Py=None, divideHerPy=False, etavals_frac=None, ptvals_frac=None):
        ''' Compute Eq. (26) in Sec. 7 of arXiv:1607.03663 at etavals and ptvals.
        Use flavor fractions splines given in Efracspline, Efracspline2.
        `divideHerPy` == True: calculate the Herwig/Pythia ratio; False: calculate the Herwig/Pythia difference
        `start_from_ratios` == True: resum the Herwig/Pythia ratios to the average of the flavor content;
                            False: do as in Eq. (26)
        '''
        ptvals, etavals = convert_to_np_array(ptvals), convert_to_np_array(etavals)
        pt_min, pt_max = Efracspline_Her.get_limits(etavals)
        if not Efracspline_Her==None:
            pt_min2, pt_max2 = Efracspline_Py.get_limits(etavals)
        else:
            pt_min2, pt_max2 = pt_min, pt_max

        # pt_min3, pt_max3 = self.get_evaluator_limits_all_flav(etavals)
        pt_min_tot = np.max([pt_min, pt_min2, [self.ptmin_global]*len(pt_min)],axis=0)
        pt_max_tot = np.min([pt_max, pt_max2, [self.ptmax_global]*len(pt_min)],axis=0)
        ptvals = np.clip(ptvals, pt_min_tot, pt_max_tot)
        if etavals_frac==None or ptvals_frac==None:
            etavals_frac = etavals
            ptvals_frac = ptvals

        return self.resum_to_mix(etavals, ptvals, Efracspline_Her, Efracspline_Py, etavals_frac, ptvals_frac )


from coffea.lookup_tools import extractor

@dataclass
class CorrectionEvaluator:
    binning: str = "Win14"
    eta_binning_str: str = field(init=False, repr=True)
    flavors: str = np.array([])
    correction_txt_dir: str = None
    correction_txt: str = None
    inverse: bool = False,
    use_corrections: str = 'J'
    evaluator: evaluator = field(init=False, repr=False)
    ptmin_global: float = 30
    ptmax_global: float = 500
    _jetetabins: JetEtaBins = field(init=False, repr=False)

    def __post_init__(self):
        binning = self.binning
        if not binning in JERC_Constants.StrToBinsDict().keys():
            raise TypeError(f"The provided eta binning, {binning} not defined in the `common_binning` file. The available binnings are {JERC_Constants.StrToBinsDict().keys()}")
        self.eta_binning_str = '_'+binning if binning != "HCalPart" else ''

        corr_loc_Sum20_Py = [f"* * {self.correction_txt_dir+self.correction_txt+self.eta_binning_str+'.txt'}"]
        corr_loc_Sum20_Her = [f"* * {self.correction_txt_dir+self.correction_txt+'_Her'+self.eta_binning_str+'.txt'}"]

        ext = extractor()
        ext.add_weight_sets(corr_loc_Sum20_Py+corr_loc_Sum20_Her)
        ext.finalize()
        self.evaluator = ext.make_evaluator()

    def evaluate(self, etavals, ptvals, sample='Her', flavor='b'):
        samp = '_Her' if sample=='Her' else ''         
        eva = self.evaluator[f'{self.correction_txt}{samp}{self.eta_binning_str}_{flavor}{self.use_corrections}']
        if self.inverse==True:
            return eva(etavals, ptvals)
        else:
            return 1/eva(etavals, ptvals)

    def resum_to_mix_one_flav(self, etavals, ptvals, Efracspline, samp, flav, etavals_frac, ptvals_frac):
        response = self.evaluate(etavals, ptvals, samp, flav)
        return response*Efracspline.evaluate(flav, etavals_frac, ptvals_frac)

    def resum_to_mix(self, etavals, ptvals, Efracspline, etavals_frac, ptvals_frac, sample='Her'):
        ''' 
        Sum up the corrections from the fit according to the flavor fractions in `Efracspline`.
        samp = 'Her' or 'Py' for Herwig corrections or for Pythia corrections
        '''
        spline_sum = sum(
                self.resum_to_mix_one_flav(etavals, ptvals, Efracspline, sample, flav, etavals_frac, ptvals_frac )   for flav in self.flavors
            )
        return spline_sum.T
    
    def get_limits(self, sample, flavor, etavals):
        '''Get the intersect of the pt ranges of all the evaluators for the given etavals
            samp = 'Her' or 'Py'
            flav = flavor
            use_corrections = 'J' for dijet or 'T' for ttbar
            '''
        samp = '_Her' if sample=='Her' else ''         
        eva = self.evaluator[f'{self.correction_txt}{samp}{self.eta_binning_str}_{flavor}{self.use_corrections}']
        corr_etabins = eva._bins['JetEta']
        corr_bin_idx = np.searchsorted(corr_etabins, etavals, side='right')-1
        corr_bin_idx[corr_bin_idx>=len(corr_etabins)] = len(corr_etabins)-1

        ptmins = list(eva._eval_clamp_mins.values())[0]
        ptmaxs = list(eva._eval_clamp_maxs.values())[0]
        ptmins = np.array(ptmins)[corr_bin_idx, 0]
        ptmaxs = np.array(ptmaxs)[corr_bin_idx, 0]
        return ptmins, ptmaxs
    
    def get_limits_all_flav(self, etavals):
        ''' Get the intersect of the pt ranges of all the evaluators for the given etavals for all flavors and for Herwig and Pythia evaluators. 
         '''
        lims = np.array([self.get_limits(sample='Her', flavor=flav, etavals=etavals) for flav in self.flavors])
        ptmins = lims[:,0,:]
        ptmaxs = lims[:,1,:]
        ptmins = np.max(ptmins, axis=0)
        ptmaxs = np.min(ptmaxs, axis=0)
        lims = np.array([self.get_limits(sample='Py', flavor=flav, etavals=etavals) for flav in self.flavors])
        ptmins = np.max(np.vstack([ptmins, lims[:,0,:]]),axis=0)
        ptmaxs = np.min(np.vstack([ptmaxs, lims[:,1,:]]),axis=0)
        return ptmins, ptmaxs

    def resum_ratio_to_mix(self, etavals, ptvals, Efracspline_Her, Efracspline_Py=None, divideHerPy=False, etavals_frac=None, ptvals_frac=None):
        ''' Compute Eq. (26) in Sec. 7 of arXiv:1607.03663 at etavals and ptvals.
        Use flavor fractions splines given in Efracspline, Efracspline2.
        `divideHerPy` == True: calculate the Herwig/Pythia ratio; False: calculate the Herwig/Pythia difference
        `start_from_ratios` == True: resum the Herwig/Pythia ratios to the average of the flavor content;
                            False: do as in Eq. (26)
        '''
        ptvals, etavals = convert_to_np_array(ptvals), convert_to_np_array(etavals)
        pt_min, pt_max = Efracspline_Her.get_limits(etavals)
        if not Efracspline_Her==None:
            pt_min2, pt_max2 = Efracspline_Py.get_limits(etavals)
        else:
            pt_min2, pt_max2 = pt_min, pt_max

        pt_min3, pt_max3 = self.get_limits_all_flav(etavals)
        pt_min_tot = np.max([pt_min, pt_min2, pt_min3, [self.ptmin_global]*len(pt_min)],axis=0)
        pt_max_tot = np.min([pt_max, pt_max2, pt_max3, [self.ptmax_global]*len(pt_min)],axis=0)
        ptvals = np.clip(ptvals, pt_min_tot, pt_max_tot)
        if etavals_frac==None or ptvals_frac==None:
            etavals_frac = etavals
            ptvals_frac = ptvals

        return get_ratio(self.resum_to_mix(etavals, ptvals, Efracspline_Her, etavals_frac, ptvals_frac, 'Her'),
                         self.resum_to_mix(etavals, ptvals, Efracspline_Py, etavals_frac, ptvals_frac, 'Py'),
                         divideHerPy)

def get_additional_uncertainty_curves(etavals, ptvals, etavals0, dijetat_eta0, evaluator, qfrac_spline_dict, divideHerPy=False):
    '''Obtain all the other curves necessary for the plots'''
    result = {}
    result["g20q80"] = evaluator.resum_ratio_to_mix(etavals, ptvals,
                                qfrac_spline_dict['DY-MG-Her'],
                                qfrac_spline_dict['DY-MG-Py'],
                                divideHerPy,
                                0, 200
                                         )

    result["g20q80_fixed"] = evaluator.resum_ratio_to_mix(0, 200,
                                          qfrac_spline_dict['DY-MG-Her'],
                                          qfrac_spline_dict['DY-MG-Py'],
                                          divideHerPy,
                                          )

    ### create custom FlavorFractions objects of 100% of a given flavor
    pt_bins = PtBins("MC_truth")
    one_spline = get_spline(np.array([1]*pt_bins.nbins), pt_bins)
    zero_spline = get_spline(np.array([1e-15]*pt_bins.nbins), pt_bins)
    eta_binning = qfrac_spline_dict['DY-MG-Her'].binning
    flavors = qfrac_spline_dict['DY-MG-Her'].flavors
    jeteta_bins = JetEtaBins(eta_binning, absolute=True)
    # pt_bins = PtBins("MC_truth", centres=qfrac_spline_dict['DY-MG-Her'].get_x())

    for flav in evaluator.flavors:
        fractions100_tmp = {flavii: np.array([zero_spline]*jeteta_bins.nbins) for flavii in flavors if flav not in flavii}
        fractions100_tmp[flav] = np.array([one_spline]*jeteta_bins.nbins)
        fractions100 = FlavorFractions(fractions100_tmp, eta_binning)
        
        result[flav+'100'] = evaluator.resum_ratio_to_mix(etavals, ptvals,
                              fractions100,
                              fractions100,
                              divideHerPy,
                             )
        
    
    Rdijet0 = evaluator.resum_ratio_to_mix(etavals0, ptvals,
                                  qfrac_spline_dict['QCD-MG-Her'],
                                  qfrac_spline_dict['QCD-MG-Py'],
                                  divideHerPy,
                                  )
    
    result["Rref"] = result["g20q80_fixed"] + (dijetat_eta0 - Rdijet0) #HerPy_differences['QCD'][0]
    
    return result
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

# #resum_to_mix(self, etavals, ptvals, Efracspline, Efracspline2, etavals_frac, ptvals_frac ):
