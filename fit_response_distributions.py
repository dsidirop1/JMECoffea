#!/usr/bin/env python
    # coding: utf-8

import sys
coffea_path = '/afs/cern.ch/user/a/anpotreb/top/JERC/coffea/'
if coffea_path not in sys.path:
    sys.path.insert(0,coffea_path)

# ak_path = '/afs/cern.ch/user/a/anpotreb/top/JERC/local-packages/'

# if ak_path not in sys.path:
#     sys.path.insert(0,ak_path)

from coffea import util
import numpy as np

# import inspect
import matplotlib.pyplot as plt
import hist
import warnings

from plotters.pltStyle import pltStyle
pltStyle(style='hep') #, font_frac=1.40
plt.rcParams['figure.subplot.left'] = plt.rcParams['figure.subplot.left']*1.3
# plt.rcParams['font.size'] = plt.rcParams['font.size']/0.98
plt.rcParams['figure.dpi'] = 150
import os

### import subpackages
import helpers as h
import plotters.plotters as plotters
# from common_binning import JERC_Constants
from fileNames.available_datasets import dataset_dictionary
from plotters.plot_cutflow import plot_cutflow

def main(data_tag='Pythia-TTBAR'):
    # The script fits the response histograms (or calculates the medians) and creates the `txt` files with the fit results (one fle for each `Mean`, `MeanStd`, `Median`, `MedianStd`, `MeanRecoPt`)

    ################ Parameters of the run and switches  #########################
    test_run            = False   ### True check on a file that was created with a processor with `test_run=True` (maybe obsolete because this can be specified just in the data_tag)
    load_fit_res        = False   ### True if only replot the fit results without redoing histogram fits (also kind of obsolete because plotting scripts exist in `plotters` )
    saveplots           = False    ### True if save all the response distributions. There are many eta/pt bins so it takes time and space
    combine_antiflavour = True    ### True if combine the flavor and anti-flavour jets into one histogram
    
    ### Choose eta binning for the response fits.
    ### HCalPart: bin in HCal sectors, CaloTowers: the standard JERC binning,
    ### CoarseCalo: like 'CaloTowers' but many bins united; onebin: combine all eta bins
    ### Preprocessing always done in CaloTowers. For the reponse distributions, the bins can be merged.
    eta_binning  = "HCalPart"  ### HCalPart, CoarseCalo, JERC, CaloTowers, Summer20Flavor, onebin;
    sum_neg_pos_eta_bool=True  ### if combining the positive and negative eta bins
    tag_Lx = '_L5'                 ### L5 or L23, but L23 not supported since ages.
    
    ### Define the dataset either by using a `data_tag` available in `dataset_dictionary`
    ### Or manualy by defining `dataset` (below) with the path to the .txt file with the file names (without the redirectors).
    ### Or manually by defining `fileslist` as the list with file names.
    ### data_tag will be used to name output figures and histograms.
    # data_tag = 'Herwig-TTBAR' # 'QCD-MG-Her' #'Herwig-TTBAR' 
    # data_tag = 'DY-FxFx'
    ### name of the specific run if parameters changed used for saving figures and output histograms.
    add_tag:str = ''   
    ### if the fit strategy changed and the results need to be stored with a different name
    fit_tag:str = ''   


    ### Define which flavors should be fit
    flavors = ['b', 'ud', 'all', 'g', 'c', 's', 'q', 'u', 'd', 'unmatched']
    # flavors = ['all']

    ### None if all the pt bins should be fit, otherwise a list of two numbers for the range of pt bins to fit, or just one number for a single pt bin
    pt_to_fit: list = None
    # pt_to_fit: list = [30]
    eta_to_fit: list = None

    ################ End of the parameters of the run and switches  #########################
    # ### Do some logic with the input partameters and the rest of parameters of the run
    
    tag_full = tag_Lx+'_'+data_tag+add_tag
    if test_run:
        tag_full = tag_full+'_test'
    outname = 'out/CoffeaJERCOutputs'+tag_full+'.coffea'
    
    tag_fit_res = tag_full
    
    if eta_binning != "HCalPart":
        tag_fit_res=tag_full+'_'+eta_binning
    combine_antiflavour_txt = '_split_antiflav' if not combine_antiflavour else ''
    tag_fit_res += combine_antiflavour_txt+fit_tag

    if not os.path.exists("out"):
        os.mkdir("out")
            
    if not os.path.exists("fig"):
        os.mkdir("fig/")
        os.mkdir("fig/responses/")
        
    if test_run and not os.path.exists("test"):
        os.mkdir("test/")
        os.mkdir("test/fig")
        
    out_txt_path = "out_txt" if not test_run else "test/out_txt"
    if not os.path.exists(out_txt_path):
        os.mkdir(out_txt_path)

    # ### End of do some logic with the input partameters and the rest of parameters of the run
    ################ Load the histograms and scale them according to their cross-sections  #########################
    output = util.load(outname)
    print("Loaded histograms from: ", outname)
    xsec_dict, legend_label = h.get_xsec_dict(data_tag, dataset_dictionary)
    
    keys = output.keys()
    try: ## in older files cutflow was not split into cutflow for jets and events. Should be removed in the future
        Nev = {key: output[key]['cutflow_events']['all_events'].value for key in keys}
    except KeyError:
        Nev = {key: output[key]['cutflow']['all_events'].value for key in keys}
    scale_factors = h.hist_div(xsec_dict, Nev)
    all_histo_keys = output[next(iter(output.keys()))].keys()
    hists_merged = {histo_key:h.sum_subhist(output, histo_key, scale_factors) for histo_key in all_histo_keys }
    
    # ### Fit responses
    
    # Define some global variables for the fit
    from JetEtaBins import JetEtaBins, PtBins
    
    jeteta_bins = JetEtaBins(eta_binning)
    pt_bins = PtBins("MC_truth")
    fiteta_bins = JetEtaBins(eta_binning, absolute=True) if sum_neg_pos_eta_bool else jeteta_bins

    if pt_to_fit is None:
        pt_bins_to_fit = range(pt_bins.nbins)
    elif len(pt_to_fit)==1:
        pt_bins_to_fit = [pt_bins.get_bin_idx(pt_to_fit[0])]
    elif len(pt_to_fit)==2:
        pt_bins_to_fit = range(pt_bins.get_bin_idx(pt_to_fit[0]), pt_bins.get_bin_idx(pt_to_fit[1])+1)

    if eta_to_fit is None:
        eta_bins_to_fit = range(fiteta_bins.nbins)
    elif len(eta_to_fit)==1:
        eta_bins_to_fit = [fiteta_bins.get_bin_idx(eta_to_fit[0])]
    elif len(eta_to_fit)==2:
        eta_bins_to_fit = range(fiteta_bins.get_bin_idx(eta_to_fit[0]), fiteta_bins.get_bin_idx(eta_to_fit[1])+1)
    
    def fit_responses(hists, flavor='b', saveplots = None, scaled_hist=None):
        ''' Extract the jet flavor `flavor` from the histogram dictionary `hists` and fit in all the eta and pt bins.
        Add `scaled_hist` if to produce the response distributions with all the samples stacked up.
        Return a dictionary of ["Mean", "MeanStd", "Median", "MedianStd", "MeanRecoPt"] values.
        
        '''
        warnings.filterwarnings('ignore')  ### filter out the many fit warnings
        if saveplots==None:
            saveplots = False if test_run or eta_binning != "HCalPart" else True
            
        response_hists = {}
        recopt_hists = {}
        if not scaled_hist==None:
            for sample in scaled_hist:
                response_hist, recopt_hist = h.add_flavors(scaled_hist[sample], flavor, combine_antiflavour) 
                response_hist = h.rebin_hist(response_hist, 'jeteta' , jeteta_bins.edges)
                recopt_hist = h.rebin_hist(recopt_hist, 'jeteta' , jeteta_bins.edges)
                if sum_neg_pos_eta_bool==True:
                    response_hist = h.sum_neg_pos_eta(response_hist)
                    recopt_hist = h.sum_neg_pos_eta(recopt_hist)
                response_hists[sample] = response_hist
                recopt_hists[sample] = recopt_hist
            
        response_hist, recopt_hist = h.add_flavors(hists, flavor, combine_antiflavour)
        
        response_hist = h.rebin_hist(response_hist, 'jeteta' , jeteta_bins.edges)
        recopt_hist = h.rebin_hist(recopt_hist, 'jeteta' , jeteta_bins.edges)
        
        if sum_neg_pos_eta_bool==True:
            response_hist = h.sum_neg_pos_eta(response_hist)
            recopt_hist = h.sum_neg_pos_eta(recopt_hist)
            
        results = {key:np.zeros((pt_bins.nbins, fiteta_bins.nbins))
                      for key in ["Mean", "MeanStd", "Median", "MedianStd", "MeanRecoPt"]  }
                                
        N_converge = 0
        N_not_converge = 0
    
        FitFigDir1 = 'fig/responses/responses'+tag_full
        if saveplots and not os.path.exists(FitFigDir1):
            os.mkdir(FitFigDir1)
        
        FitFigDir = FitFigDir1+'/response_pt_eta_'+flavor+tag_full
        if saveplots:
            if not os.path.exists(FitFigDir):
                os.mkdir(FitFigDir)
            print("Response fits will be saved under ", FitFigDir)
        elif not saveplots:
            print("Response fits won't be saved")
    
        for i in pt_bins_to_fit:
            for k in eta_bins_to_fit:
                if not scaled_hist==None:
                    histos = {sample: response_hists[sample][i, :, k] for sample in response_hists}
                    histos2plot = {key[10:]:histos[key] for key in histos.keys()}
                    h_stack = hist.Stack.from_dict(histos2plot)
                
                histo = response_hist[i, :, k]
                histopt = recopt_hist[i, k]            
                try:
                    Neff = histo.sum().value**2/(histo.sum().variance)
                except ZeroDivisionError:
                    Neff = histo.sum().value**2/(histo.sum().variance+1e-20)
    
                median, medianstd = h.get_median(histo, Neff) #, x_range=[0, 2]
                
                ##################### Mean of the pt_reco  ######################
                ### (The mean includes events that potentially had ptresponse in the second peak at low pt)
                ### No way to distinguish it if only x*weights are saved instead of the whole histogram.
                mean_reco_pt = histopt.value/np.sum(histo.values())
                
                ####################### Fitting ############################
                p2, cov, chi2, Ndof, if_failed, fitlims = h.fit_response(histo, Neff, Nfit=3, sigma_fit_window=1.5)
                if if_failed:
                    N_not_converge += 1
                else:
                    N_converge += 1
                
                ####################### Store the results ############################
                results["Mean"][i,k] = p2[1]
                results["MeanStd"][i,k] = np.sqrt(np.abs(cov[1,1]))
                results["Median"][i,k] = median
                results["MedianStd"][i,k] = medianstd
                results["MeanRecoPt"][i,k] = mean_reco_pt
    
        ####################### Plotting ############################
                if  saveplots:
                    figName = FitFigDir+'/ptResponse'+pt_bins.idx2str(i)+fiteta_bins.idx2str(k)
                    hep_txt = pt_bins.idx2plot_str(i)+'\n'+fiteta_bins.idx2plot_str(k)+'\n'+f'{flav} jet' 
            
                    txt2print = ('\n'+r'Mean = {0:0.3f}$\pm${1:0.3f}'.format(p2[1], np.sqrt(cov[1,1]))
                                     + '\nWidth = {0:0.3f}$\pm${1:0.3f}'.format(np.abs(p2[2]), np.sqrt(cov[2,2]))
                                     + '\n'+r'Median = {0:0.3f}$\pm${1:0.3f}'.format(median, medianstd)
                                     + '\n'+r'$\chi^2/ndof$ = {0:0.2g}/{1:0.0f}'.format(chi2, Ndof)
                                     + '\n'+r'Neff = {0:0.3g}'.format(Neff))
                    plotters.plot_response_dist(histo, p2, fitlims,
                                       figName, dataset_name=legend_label, hep_txt=hep_txt, txt2print=txt2print, print_txt=True)              
                    if not scaled_hist==None:
                        plotters.plot_response_dist_stack(h_stack, p2, fitlims,
                                                 figName+'stack', hep_txt=hep_txt, print_txt=False )
    
        print("N converge = ", N_converge, "N_not_converge = ", N_not_converge );
        warnings.filterwarnings('default')
        
        return results  
    
    # ### Run fitting for each sample
    
    medians = []
    medianstds = []
    if not combine_antiflavour:
        flavors = np.concatenate([[flav, flav+'bar'] if flav in h.barable_flavors else [flav] for flav in flavors ])
    print('-'*25)
    print('-'*25)
    print(f'Starting to fit each flavor in: {flavors}')
    for flav in flavors:
        print('-'*25)
        print('-'*25)
        print('Fitting flavor: ', flav)
        if load_fit_res:
            result = {}
            keys = ["Mean", "MeanStd", "Median", "MedianStd", "MeanRecoPt"] 
            for key in keys:
                result[key] = h.read_data(key, flav, tag_fit_res)
        
        else:
            result = fit_responses(hists_merged, flav, saveplots=saveplots) #scaled_hist
            medians.append(result["Median"][0][0])
            medianstds.append(result["MedianStd"][0][0])
            for key in result:
                h.save_data(result[key], key, flav, tag_fit_res, pt_bins.centres, fiteta_bins.edges, out_txt_path)
                pass
                
    #     print("result = ", result)
        # median = result["Median"]
        # medianStd = result["MedianStd"] 
        # meanstd = np.sqrt(result["MeanStd"])
                
        if eta_binning=="one_bin": #or fine_etabins:
            plotters.plot_corrections_eta(result["Median"], result["MedianStd"], pt_bins, fiteta_bins.centres, tag_fit_res, flav, plotptvals=[20, 35, 150, 400])
        else:
            plotters.plot_corrections(result, pt_bins.centres, fiteta_bins, tag_fit_res, flav, plotetavals=[0, 1.305, 2.5, 3.139], plotmean=True)
    #         plotters.plot_corrections_eta(result["Median"], result["MedianStd"], pt_bins, fiteta_bins.centres, tag_fit_res, flav, plotptvals=[20, 35, 150, 400])
    

    print('-'*25)
    print('-'*25)
    print('Saving cutflow')
    rc_bottom_def = plt.rcParams['figure.subplot.bottom']
    plt.rcParams['figure.subplot.bottom'] = 0.39
    tag_cutflow = tag_full[4:]
    hist1 = output[list(keys)[0]] ### plotting only the first
    if 'cutflow_events' in hist1:
        ## drawing only for the first sample as in the hist_merged, the total number of events are normalized
        plot_cutflow(hist1['cutflow_events'], list(keys)[0], ylab='N events', fig_name='cutflow_Nevents')
        plot_cutflow(hist1['cutflow_jets'], list(keys)[0], ylab='N jets', fig_name='cutflow_Njets')
        plot_cutflow(hists_merged['cutflow_jets'], tag_cutflow, ylab='N jets/N events', fig_name='cutflow_Njets_per_ev')
    else:
        print("cutflow histograms cannot be drawn because the cutflow isn't split into events and jets. Potentially not all events are selected")
        cutflow = hist1['cutflow'][['all_events', 'selected_events',  'events passing the lepton selection',  'events, alpha cut' ]]
        plot_cutflow(cutflow, list(keys)[0], ylab='N events', fig_name='cutflow_Nevents')

        cutflow = hist1['cutflow'][[ 'all_jets', 'gen_matched', 'jets, tight lepton id',
                                        'jets, dR cut with leptons', 'jetpt cut',
                                        'alpha cut; leading jets','iso jets']]
        plot_cutflow(cutflow, list(keys)[0], ylab='N jets', fig_name='cutflow_Njets')
    plt.rcParams['figure.subplot.bottom'] = rc_bottom_def
    print('-----'*10)
    print("All done. Congrats!")
  
if __name__ == "__main__":
    # data_tags = ['Pythia-TTBAR', 'Herwig-TTBAR', 'QCD-MG-Py', 'QCD-MG-Her', 'QCD-Py', 'DY-MG-Py', 'DY-MG-Her']
    # data_tags = ['Pythia-TTBAR_iso_dr_0p8','Pythia-TTBAR_iso_dr_1p2', 'Pythia-TTBAR_iso_dr_1p5'] #Pythia-semilep-TTBAR
    # data_tags = ['scaled_times2_pion', 'scaled_times5_pion', 'scaled_times10_pion', 'scaled_pion', 'not_scaled_pion'] #Pythia-semilep-TTBAR
    # data_tags = ['scaled_times2_pion', 'scaled_times5_pion', 'scaled_times10_pion', 'scaled_pion', 'not_scaled_pion'] #Pythia-semilep-TTBAR

    data_tags = ['QCD-Py_leading_gen_jet',] # 'Pythia-non-semilep-TTBAR', 'DY-MG-Py', 'QCD-MG-Py' Pythia-semilep-TTBAR
    # data_tags = ['QCD-Py', 'Pythia-semilep-TTBAR', 'DY-MG-Py' ] #Pythia-semilep-TTBAR
    for data_tag in data_tags:
        main(data_tag=data_tag)