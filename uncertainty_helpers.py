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
    Neff = {key: output[key]['cutflow_events']['all_events'].value if 'genwt' not in data_tag else output[key]['sum_weights']['sum_weights'].value for key in keys}
    # Nev = {key: output[key]['cutflow_events']['all_events'].value for key in keys}
    # response_sums = {key:sum(dictionary_pattern(output[key], "ptresponse_").values()).sum().value for key in output.keys()}
    # scale_factors = {key:1 for key in output.keys()} #hist_div(xsec_dict, Nev)
    scale_factors = hist_div(xsec_dict, Neff)
    all_histo_keys = output[next(iter(output.keys()))].keys()
    hists_merged = {histo_key:sum_subhist(output, histo_key, scale_factors) for histo_key in all_histo_keys }  
    return hists_merged

def combine_flavors(output, flavors, sumeta=True, combine_antiflavour=True):
    ''' Add the flavors of the histograms in `output` according to the 'helpers.composite_sample_dict' and sum over response axis.
    sumeta: if true, sum over eta axis, otherwise sum only over the response axis
    combine_antiflavour: if true, combine the antiflavours (e.g. combine b and bbar in one histogram)
    Return a dictionary over the flavors
    '''
    hists = {}
    for flav in flavors:
        combined = add_flavors(output, flavor=flav, combine_antiflavour=combine_antiflavour )[0]
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
ptmax_global = 1000

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

def create_data_HerPy_differences(sampHer, sampPy, pt_idxs, eta_idxs,
                                  inverse=False, divideHerPy=False,
                                  eta_binning_str='Summer20Flavor',
                                  ):
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