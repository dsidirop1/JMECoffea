import os
from helpers import read_data
from fit_response_distributions import fit_response_distributions
# import subprocess
import numpy as np

def match_to_filename(keys, data_tag):
    matched =  np.where([ key in data_tag for key in keys])[0]
    if len(matched)>1:
        raise ValueError(f"More than one key in [{keys}] matches the given data_tag = {data_tag}")
    elif len(matched)==1:
        return keys[matched[0]]

# def match_to_filename(eta_keys, pt_keys, data_tag):
#     matched =  np.where([ key in data_tag for key in keys])[0]
#     if len(matched)>1:
#         raise ValueError(f"More than one key in [{keys}] matches the given data_tag = {data_tag}")
#     elif len(matched)==1:
#         return keys[matched[0]]

def make_fit_config(tag):
    eta_match = match_to_filename(['_CoarseCalo', '_JERC', '_CaloTowers', '_Summer20Flavor', '_onebin'], tag)
    if eta_match is None:
        eta_binning = "HCalPart"
    else:
        eta_binning = eta_match[1:]
        tag = tag.replace(eta_match, '')

    pt_match = match_to_filename(['_pt-Uncert', '_pt-Coarse', '_pt-onebin'], tag)
     
    if pt_match is None:
        pt_binning = "MC_truth"
    else:
        pt_binning = pt_match[4:]
        tag = tag.replace(pt_match, '')
    
    if '_split_antiflav' in tag:
        combine_antiflav = False
        tag = tag.replace('_split_antiflav', '')
    else:
        combine_antiflav = True
    
    config = {
        "test_run"            : False,   ### True check on a file that was created with a processor with `test_run=True` (maybe obsolete because this can be specified just in the data_tag)
        "load_fit_res"        : False,   ### True if only replot the fit results without redoing histogram fits (also kind of obsolete because plotting scripts exist in `plotters` )
        "saveplots"           : False,    ### True if save all the response distributions. There are many eta/pt bins so it takes time and space
        "combine_antiflavour" : combine_antiflav,    ### True if combine the flavor and anti-flavour jets into one histogram
        
        ### Choose eta binning for the response fits.
        ### HCalPart: bin in HCal sectors, CaloTowers: the standard JERC binning,
        ### CoarseCalo: like 'CaloTowers' but many bins united; onebin: combine all eta bins
        ### Preprocessing always done in CaloTowers. For the reponse distributions, the bins can be merged.
        "eta_binning"         : eta_binning,
        "pt_binning"          : pt_binning, ### MC_truth, Uncert, Coarse, onebin
        "sum_neg_pos_eta_bool": True,  ### if combining the positive and negative eta bins
        "tag_Lx" : '_L5' if '_L5' in tag else '_L23',                 ### L5 or L23, but L23 not supported since ages.
        
        ### name of the specific run if parameters changed used for saving figures and output histograms.
        "add_tag":             '',   
        ### if the fit strategy changed and the results need to be stored with a different name
        "fit_tag":              '',   

        ### Define which flavors should be fit
        "flavors":                ['b', 'ud', 'all', 'g', 'c', 's', 'q', 'u', 'd', 'unmatched'],

        ### None if all the pt bins should be fit, otherwise a list of two numbers for the range of pt bins to fit, or just one number for a single pt bin
        "pt_to_fit": None,
        "eta_to_fit": None,
    }

    if '_L5' in tag:
        tag = tag.replace('_L5_', '')

    print(f'Automatically determined a config from a data_tag: {config}')
    print('The cleaned tag is: ', tag)
    return config, tag


def read_or_recreate_data(name, flavor, tag, path='out_txt'):
    file_path = path+'/EtaBinsvsPtBins'+name+'_'+flavor+tag+'.csv'
    if not os.path.exists(file_path):
        config, cleaned_tag = make_fit_config(tag)
        # breakpoint()
        if os.path.exists(path+f'/../out/CoffeaJERCOutputs_L5_{cleaned_tag}.coffea'):
            print(f"The text file with fit results {file_path} does not exist, but the output histograms in {path+f'/../out/CoffeaJERCOutputs_L5_{cleaned_tag}.coffea'} do exist.")
            create_file = input("Do you want to create the fit results? (yes/no): ")
            if create_file.lower() == 'yes' or create_file.lower() == '' or create_file.lower() == 'y':
                fit_response_distributions(cleaned_tag, config=config)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File creation failed. The file {file_path} does not exist. The file tag used {tag}")
            else:
                raise FileNotFoundError(f"The file {file_path} does not exist. The file tag used {tag}")
        else:
            raise FileNotFoundError(f"The file {file_path} does not exist. The file tag used {tag}")

    return read_data(name, flavor, tag, path=path)