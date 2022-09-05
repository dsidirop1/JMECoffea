def main():
    
    import bokeh
    import time
    import copy
    import scipy.stats as ss
    from scipy.optimize import curve_fit
    from coffea import hist, processor, nanoevents, util
    from coffea.nanoevents.methods import candidate
    from coffea.nanoevents import NanoAODSchema, BaseSchema
    
    import awkward as ak
    import numpy as np
    import glob as glob
    import itertools
    import pandas as pd
    from numpy.random import RandomState
    
    from dask.distributed import Client
    import inspect
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    from pltStyle import pltStyle
    import os
    
    from CoffeaJERCProcessor_L5 import Processor
    
    UsingDaskExecutor = True
    CoffeaCasaEnv     = False
    load_preexisting  = False    ### True if don't repeat the processing of files and use preexisting JER from output
    test_run          = False     ### True if run only on one file
    load_fit_res      = False
    
    tag = '_L5'
    
    exec('from CoffeaJERCProcessor'+tag+' import Processor') 
    
    add_tag = '_QCD' # '_TTBAR' #'_QCD' # '_testing_19UL18' # ''
    tag_full = tag+add_tag
    
    outname = 'out/CoffeaJERCOutputs'+tag_full+'.coffea'
    outname = outname+'_test' if test_run else outname
    
    if load_preexisting == True:
        UsingDaskExecutor = False
        
    if not os.path.exists("out"):
        os.mkdir("out")
        
    if not os.path.exists("out_txt"):
        os.mkdir("out_txt")
        
    if test_run and not os.path.exists("test"):
        os.mkdir("test/")
        os.mkdir("test/out_txt")
        os.mkdir("test/fig")
    
    xrootdstr = 'root://cmsxrootd.fnal.gov/'
    
    if CoffeaCasaEnv:
        xrootdstr = 'root://xcache/'
        
    dataset = 'fileNames_QCD20UL18.txt'
    
    rootfiles = open(dataset).read().split()
    
    fileslist = [xrootdstr + file for file in rootfiles]
    fileslist = fileslist[:40] # if add_tag=='QCD' else fileslist # :20 to skim the events
    
    fileslist = fileslist
    if test_run:
        fileslist = [fileslist[0]]
        ### The smallest file in the RunIISummer20UL18NanoAODv9 dataset
    
    import os
    
    os.environ['X509_USER_PROXY'] = '/afs/cern.ch/user/a/anpotreb/k5-ca-proxy.pem'
    if os.path.isfile(os.environ['X509_USER_PROXY']):
        print("Found proxy at {}".format(os.environ['X509_USER_PROXY']))
    else:
        print("os.environ['X509_USER_PROXY'] ",os.environ['X509_USER_PROXY'])
    os.environ['X509_CERT_DIR'] = '/cvmfs/cms.cern.ch/grid/etc/grid-security/certificates'
    os.environ['X509_VOMS_DIR'] = '/cvmfs/cms.cern.ch/grid/etc/grid-security/vomsdir'
    
    import uproot
    
    ff = uproot.open(fileslist[0])
    ff.keys()
    ff.close()
    
    fileslist[0]
    
    import awkward as ak
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    
    from coffea.jetmet_tools import JetCorrectionUncertainty #FactorizedJetCorrector
    from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
    
    from coffea.lookup_tools import extractor
    
    events = NanoEventsFactory.from_root(
        'root://cmsxrootd.fnal.gov///store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/87DEE912-70CF-A549-B10B-1A229B256E88.root', #fileslist[0],
        schemaclass=NanoAODSchema.v6,
        entry_stop=10
    ).events()
    jets = events.Jet
    
    jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
    jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
    jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll[:10], jets['pt'])[0]
    
    ext = extractor()
    ext.add_weight_sets([
        "* * Summer20UL18_V2_MC/Summer20UL18_V2_MC_L1FastJet_AK4PFchs.txt",
        "* * Summer20UL18_V2_MC/Summer20UL18_V2_MC_L2Relative_AK4PFchs.txt",
        "* * Summer20UL18_V2_MC/Summer20UL18_V2_MC_L3Absolute_AK4PFchs.txt",
    ])
    ext.finalize()
    
    jec_stack_names = ["Summer20UL18_V2_MC_L1FastJet_AK4PFchs",
                       "Summer20UL18_V2_MC_L2Relative_AK4PFchs", 
                       "Summer20UL18_V2_MC_L3Absolute_AK4PFchs",
                      ]
    
    evaluator = ext.make_evaluator()
    
    print(evaluator)
    print(evaluator.keys())
    
    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    jec_stack = JECStack(jec_inputs)
    
    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'rho'
            
    events_cache = events.caches[0]
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    corrected_jets = jet_factory.build(jets, lazy_cache=events_cache)
    
    {'jet_energy_correction', 'mass_orig', 'pt_orig', 'pt_jec', 'JES_jes', 'mass_jec', 'jet_energy_uncertainty_jes'} - set(ak.fields(corrected_jets))
    
    npvCut = (events.PV.npvsGood > 0)
    pvzCut = (np.abs(events.PV.z) < 24)
    rxyCut = (np.sqrt(events.PV.x*events.PV.x + events.PV.y*events.PV.y) < 2)
    
    selectedEvents = events[npvCut & pvzCut & rxyCut]
    
    ak.num(jets)
    
    GenJets = selectedEvents.GenJet #[:,0:2]
    
    gens, dr = jets.nearest(GenJets, return_metric=True)
    mask = (dr < 0.2)
    
    jet = jets[mask]
    gen = gens[mask]
    print(gen.eta.to_list())
    match_gen1 = gen.eta.to_list()
    
    matched_gen_eta = ak.flatten(corrected_jets.matched_gen.eta).to_numpy()
    match_gen2 = corrected_jets.matched_gen.eta.to_list()
    
    matchedJets = ak.cartesian([GenJets, corrected_jets])
    deltaR = matchedJets.slot0.delta_r(matchedJets.slot1)
    matchedJets = matchedJets[deltaR < 0.2]
    aa = corrected_jets.matched_gen
    aa = aa[ak.num(aa)>0]
    jj = ak.flatten(corrected_jets.matched_gen.eta).to_list()
    
    match_gen3 = matchedJets.slot0.eta.to_list()
    jets3 = matchedJets.slot1.eta.to_list()
    
    jet_pt_mask = jets.pt>20
    jet_gen_match_mask = ~ak.is_none(corrected_jets.matched_gen,axis=1)
    dressed_electron_mask = (ak.sum(ak.is_none(jets.matched_electrons,axis=2), axis=2)==2) #no matched electron to jet
    dressed_muon_mask = (ak.sum(ak.is_none(jets.matched_muons,axis=2), axis=2)==2)
    
    drs, _ = jets.metric_table(jets, return_combinations=True, axis=1)
    drs = ak.min(drs[drs>1e-10],axis=2) #mask the drs jet back to the same jet and find the smallest
    jet_iso_mask = drs>0.8
    jet_mask = jet_pt_mask & jet_gen_match_mask & dressed_electron_mask & dressed_muon_mask & jet_iso_mask
    
    ak.sum(jet_pt_mask)
    ak.sum(jet_gen_match_mask)
    ak.sum(dressed_electron_mask)
    ak.sum(dressed_muon_mask)
    ak.sum(jet_iso_mask)
    ak.sum(jet_mask)
    
    jets2 = jets[jet_mask]
    corrected_jets.matched_gen[jet_mask]
    
    ak.flatten(jets.eta).to_numpy()
    
    ak.flatten(jets2[-5:-4].pt).to_numpy() #allow_missing=True)
    
    ak.flatten(jets2[-3:-2].eta).to_numpy(allow_missing=True)
    
    ak.flatten(jets2.eta).to_numpy(allow_missing=True)
    ak.num(jets2)
    
    events.GenPart[
        events.GenPart.hasFlags(['isPrompt', 'isLastCopy'])
    ][0].pdgId
    
    if(UsingDaskExecutor and CoffeaCasaEnv):
        client = Client("tls://ac-2emalik-2ewilliams-40cern-2ech.dask.coffea.casa:8786")
        client.upload_file('CoffeaJERCProcessor.py')
    
    if(UsingDaskExecutor and not CoffeaCasaEnv):
        
        
        from dask.distributed import Client
        client = Client()
        client.upload_file('CoffeaJERCProcessor'+tag+'.py')
    
    tstart = time.time()
    
    outputs_unweighted = {}
    
    seed = 1234577890
    prng = RandomState(seed)
    Chunk = [10000, 5] # [chunksize, maxchunks]
    
    filesets = {'QCD': fileslist}
    
    if not load_preexisting:
        for name,files in filesets.items(): 
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=Processor(),
                                                  executor=processor.iterative_executor,
            #                                        executor=processor.futures_executor,
                                                  executor_args={
                                                      'skipbadfiles':False,
                                                      'schema': NanoAODSchema, #BaseSchema
                                                      'workers': 2},
                                                  chunksize=Chunk[0])#, maxchunks=Chunk[1])
            else:
                chosen_exec = 'dask'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=Processor(),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': NanoAODSchema, #BaseSchema
        #                                               'workers': 2
                                                  },
                                                  chunksize=Chunk[0])#, maxchunks=Chunk[1])
    
        elapsed = time.time() - tstart
        outputs_unweighted[name] = output
        util.save(output, outname)
        outputs_unweighted[name] = output
        print(name + ' unweighted output loaded')
    else:
        # output = util.load('out/CoffeaJERCOutputs_binned.coffea')
        output = util.load(outname)
        
    elapsed = time.time() - tstart
    
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
    f_xvals = np.linspace(0,5,5001)
    ptbins = output['ptresponse'].axis('pt').edges()
    ptbins_c = output['ptresponse'].axis('pt').centers()
    etabins = output['ptresponse'].axis('jeteta').edges()
    jetpt_length = len(ptbins)-1
    jeteta_length = (len(etabins)-1)//2
    
    etabins_mod = etabins[(len(etabins)-1)//2:]
    
    ptresp_edd = output['ptresponse'].axis('ptresponse').edges()
    plot_pt_edges = ptresp_edd[0:np.nonzero(ptresp_edd>=2.0)[0][0]]
    hist_pt_edges = plot_pt_edges[1:-1]   #for plotting. To exclude overflow from the plot
    plot_response_axis = hist.Bin("jeteta", r"Jet $\eta$", hist_pt_edges)
    
    from pltStyle import pltStyle
    pltStyle(style='paper')
    plt.rcParams['figure.subplot.left'] = 0.162
    plt.rcParams['figure.dpi'] = 150
    
    import warnings
    
    def fit_responses(output, samp='_b'):
        warnings.filterwarnings('ignore')
    
        mean = np.zeros((jetpt_length, jeteta_length))
        medians = np.zeros((jetpt_length, jeteta_length))
        medianstds = np.zeros((jetpt_length, jeteta_length))
        width = np.zeros((jetpt_length, jeteta_length))
        meanvar = np.zeros((jetpt_length, jeteta_length))
        
        N_converge = 0
        N_not_converge = 0
    
        FitFigDir = 'fig/response_pt_eta'+samp+tag_full
        if not os.path.exists(FitFigDir):
            os.mkdir(FitFigDir)
            
        xvals = output['ptresponse'+samp].axis('ptresponse').centers()[1:] #[1:] to exclude the second peak for low pt
        f_xvals = np.linspace(0,max(xvals),5001)
        response_edges = output['ptresponse'+samp].axis('ptresponse').edges()[1:]
    
        for i in range(jetpt_length):
            ptBin = hist.Interval(ptbins[i], ptbins[i+1])
            print('-'*25)
            print('-'*25)
            
            if not 'inf' in str(ptBin):
                pt_string = '_pT'+str(int(ptBin.lo))+'to'+str(int(ptBin.hi))
            else:
                pt_string = '_pT'+str(ptBin.lo) + 'to' + str(ptBin.hi)
                pt_string = pt_string.replace('.0','').replace('-infto','0to')
    
            for k in range(jeteta_length):
                etaBinPl = hist.Interval(etabins[k+jeteta_length], etabins[k+1+jeteta_length])
                etaBinMi = hist.Interval(etabins[jeteta_length-k-1], etabins[jeteta_length-k])
                print("etaBin = ", etaBinPl ,", ptBin = ", ptBin )
                eta_string = '_eta'+str(etaBinPl.lo)+'to'+str(etaBinPl.hi)
                eta_string = eta_string.replace('.','')
                
                # The name integrate is a bit misleasding in this line. Is there another way to "slice" a histogram? //Andris
                histoMi = output['ptresponse'+samp].integrate('jeteta', etaBinMi).integrate('pt', ptBin)
                histoPl = output['ptresponse'+samp].integrate('jeteta', etaBinPl).integrate('pt', ptBin)
                histo = (histoMi+histoPl)
                yvals = histo.values()[('QCD',)][1:]  #[1:] to exclude the second peak for low pt
    
                N = histo.integrate('ptresponse').values()[('QCD',)]
                
               ####################### Calculate median and rms ############################
                yvals_cumsum = np.cumsum(yvals)
                med_bin = np.nonzero(yvals_cumsum>N/2)[0][0] if N>0 else 0
                median = response_edges[med_bin] + (N/2 - yvals_cumsum[med_bin-1])/yvals[med_bin]*(response_edges[med_bin+1]
                                                                                          - response_edges[med_bin])
                
                hist_mean = np.sum(xvals*yvals)/sum(yvals) 
                hist_rms = np.sqrt(np.sum(yvals*((hist_mean-xvals)**2))/sum(yvals))
                medianstd = 1.253 * hist_rms/np.sqrt(N)
                
               ####################### Fitting ############################
                nonzero_bins = np.sum(yvals>0)
                if nonzero_bins<2 or N<50:
                    p2=[0,0,0]
                    chi2 = np.nan
                    arr = np.array([[np.nan]*3]*3)
                    Ndof = 0
                    print("Too little data points, skipping p = ", p2)
                else:
                    try:
                        p, arr = curve_fit(gauss, xvals, yvals, p0=[10,1,1])
                        N_converge += 1
                             ######## Second Gaussian ########
                        xfit_l = np.where(xvals>=p[1]-np.abs(p[2])*1.5)[0][0]
                        xfit_h = np.where(xvals>=p[1]+np.abs(p[2])*1.5)[0][0]
                        if len(range(xfit_l,xfit_h))<6: #if there are only 3pnts, the uncertainty is infty
                            xfit_l = xfit_l-1
                            xfit_h = xfit_h+1
                            if len(range(xfit_l,xfit_h))<6:
                                xfit_l = xfit_l-1
                                xfit_h = xfit_h+1
                        xvals2 = xvals[xfit_l: xfit_h]
                        yvals2 = yvals[xfit_l: xfit_h]
                        p2, arr = curve_fit(gauss, xvals2, yvals2, p0=p)
    
                        ygaus = gauss(xvals, *p2)
                        chi2 = sum((yvals-ygaus)**2/(yvals+1E-9))
                        Ndof = len(xvals2)-3
                        if chi2<50000:
                            print("Fit converged, p = ", p2, ", chi2 = ", chi2 )
                        else:
                            print("Fit failed because of high chi2, p = ", p2, ", chi2 = ", chi2 )
                    except(RuntimeError):
                        p2=[0,0,0]
                        print("Fit failed because of non-convergance, p = ", p2)
                        chi2 = np.nan
                        arr = np.array([[np.nan]*3]*3)
                        Ndof = 0
                        N_not_converge += 1
                        continue
    
                fgaus2 = gauss(f_xvals, *p2)
    
                width_ik = np.abs(p2[2])
                
                mean[i,k] = p2[1]
                meanvar[i,k] = arr[1,1]
                medians[i,k] = median
                medianstds[i,k] = medianstd
                width[i,k] = width_ik
    
       ####################### Plotting ############################
                if not test_run:
                    histo = histo.rebin('ptresponse', plot_response_axis)
    
                    fig, ax2 = plt.subplots();
                    hist.plot1d(histo, ax=ax2, overlay='dataset', overflow='all',
                                fill_opts={'alpha': .5, 'edgecolor': (0,0,0,0.3), 'linewidth': 1.4})
                    # ax2.plot(f_xvals, fgaus, label='Gaus',linewidth=1.8)
                    ax2.plot(f_xvals, fgaus2, label='Gaus',linewidth=1.8)
                    ax2.set_xlabel("Response ($E_{RECO}/E_{GEN}$)")
                    ax2.set_xlim(plot_pt_edges[[0,-1]])
                    h = ax2.get_ylim()[1]/1.05
                    plt.text(0.03,0.95*h,r'Mean {0:0.3f}$\pm${1:0.3f}'.format(p2[1], np.sqrt(arr[1,1])))
                    plt.text(0.03,0.88*h,r'Width {0:0.3f}$\pm${1:0.3f}'.format(width_ik, np.sqrt(arr[2,2])))
                    plt.text(0.03,0.81*h,r'Median {0:0.3f}$\pm${1:0.3f}'.format(median, medianstd))
                    plt.text(0.03,0.73*h,r'$\chi^2/ndof$ {0:0.2g}/{1:0.0f}'.format(chi2, Ndof))
                    plt.text(0.03,0.66*h,r'N data = {0:0.3g}'.format(N))
                    ax2.legend();
    
                    plt.savefig(FitFigDir+'/ptResponse'+pt_string+eta_string+'.png', dpi=plt.rcParamsDefault['figure.dpi']);
                    plt.savefig(FitFigDir+'/ptResponse'+pt_string+eta_string+'.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
                    plt.close();                
    
        print("N converge = ", N_converge, "N_not_converge = ", N_not_converge );
        warnings.filterwarnings('default')
        
        return [mean, meanvar, medians, medianstds] #width, 
        
    
    df_csv = pd.read_csv('out_txt/Closure_L5_QCD.csv').set_index('etaBins')
    closure_corr = df_csv.to_numpy().transpose()
    closure_corr = np.pad(closure_corr,1,constant_values=1)
    
    legend_tmp = f'{0:0.0f}'+r'$<\eta<$'+f'{1:0.0f}'
    
    def plot_corrections(mean, samp, meanvar):
        ### To ignore the points with 0 on y axis when setting the y axis limits
        mean_p = mean.copy()
        mean_p[mean_p==0] = np.nan
    
        fig, ax = plt.subplots()
        start = 17
        
        ptbins_plot = ptbins_c[start:]
        meanstd = np.sqrt(meanvar[start:,:])
        
        k2 = np.where(etabins_mod<=0)[0][-1]
        k4 = np.where(etabins_mod<=1.3)[0][-1]
        k6 = np.where(etabins_mod<=2.5)[0][-1]
        k8 = np.where(etabins_mod<=3.0)[0][-1]
        
        plt.errorbar(ptbins_plot, mean_p[start:,k2], yerr=meanstd[:,k2], marker='o',
                     linestyle="none", label=f'{etabins_mod[k2]}'+r'$<\eta<$'+f'{etabins_mod[k2+1]}')
        plt.errorbar(ptbins_plot, mean_p[start:,k4], yerr=meanstd[:,k4], marker='o',
                 linestyle="none", label=f'{etabins_mod[k4]}'+r'$<\eta<$'+f'{etabins_mod[k4+1]}')
        plt.errorbar(ptbins_plot, mean_p[start:,k6], yerr=meanstd[:,k6], marker='o',
                 linestyle="none", label=f'{etabins_mod[k6]}'+r'$<\eta<$'+f'{etabins_mod[k6+1]}')
        plt.errorbar(ptbins_plot, mean_p[start:,k8], yerr=meanstd[:,k8], marker='o',
                 linestyle="none", label=f'{etabins_mod[k8]}'+r'$<\eta<$'+f'{etabins_mod[k8+1]}')
    
        ### Calculate resonable limits excluding the few points with insane errors
        yerr_norm = np.concatenate([np.sqrt(meanvar[start:,[k2, k4, k6, k8]]) ])
        y_norm = np.concatenate([mean_p[start:,[k2, k4, k6, k8]]])
        norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
        ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]) ,np.max((yerr_norm+y_norm)[norm_pos]))
        
        ax.set_xlabel(r'$p_T$ (GeV)');
        ax.set_ylabel(r'mean response');
        ax.set_xscale('log')
        ax.legend()
        if test_run:
            plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
            plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.png', dpi=plt.rcParamsDefault['figure.dpi']);
        else:
            plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.pdf');
            plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.png');
    
        plt.show();
        
    
        
        
        
    
        
    
        
    
    def save_data(data, name, samp):
        # data = {str(ptBin):mean[i] for i, ptBin in enumerate(output['jetpt'].axis('pt')[1:-1])}
        data_dict = {str(ptBin):data[i] for i, ptBin in enumerate(ptbins[:-1])}
    
        # data['etaBins'] = [str(etaBin) for etaBin in output['jeteta'].axis('jeteta')[1:-1]]
        data_dict['etaBins'] = np.array([str(etaBin) for etaBin in etabins_mod[:-1]])
    
        df = pd.DataFrame(data=data_dict)
        df = df.set_index('etaBins')
        if not test_run:
            df.to_csv('out_txt/EtaBinsvsPtBins'+name+samp+tag_full+'.csv')
        else:
            df.to_csv('test/out_txt/EtaBinsvsPtBins'+name+samp+tag_full+'_test.csv')
    
    def read_data(name, samp):
        if not test_run:
            df_csv = pd.read_csv('out_txt/EtaBinsvsPtBins'+name+samp+tag_full+'.csv').set_index('etaBins')
        else: #before_closure/
            df_csv = pd.read_csv('test/out_txt/EtaBinsvsPtBins'+name+samp+tag_full+'_test.csv').set_index('etaBins')
        
        data = df_csv.to_numpy().transpose()
        return data
    
                
    
    load_fit_res=False
    subsamples = ['', '_b', '_c', '_ud', '_s', '_g']
    subsamples = ['', '_ud', '_g'] # , '_b']
    for samp in subsamples:
        print('-'*25)
        print('-'*25)
        print('Fitting subsample: ', samp)
        if load_fit_res:
            mean = read_data("Mean", samp)
            meanvar = read_data("MeanVar", samp)
            median = read_data("Mean", samp)
            medianstd = read_data("MedianStd", samp)
        else:
            mean, meanvar, median, medianstd = fit_responses(output, samp)
            for data, name in zip([mean, meanvar, median, medianstd],["Mean", "MeanVar", "Median", "MedianStd"]):
                save_data(data, name, samp)
                
        plot_corrections(mean, samp, meanvar)
    
#     subsamples = ['', '_b', '_c', '_ud', '_s', '_g']
#     subsamples = ['', '_ud', '_g'] # , '_b']
#     for samp in subsamples:
#         print('-'*25)
#         print('-'*25)
#         print('Fitting subsample: ', samp)
#         if load_fit_res:
#             mean = read_data("Mean", samp)
#             meanvar = read_data("MeanVar", samp)
#             median = read_data("Mean", samp)
#             medianstd = read_data("MedianStd", samp)
#         else:
#             mean, meanvar, median, medianstd = fit_responses(output, samp)
#             for data, name in zip([mean, meanvar, median, medianstd],["Mean", "MeanVar", "Median", "MedianStd"]):
#                 save_data(data, name, samp)
                
#         plot_corrections(mean, samp, meanvar)
    
    samp = '_b'
    etabin = 1
    
    mean = read_data("Mean", samp)
    meanvar = read_data("MeanVar", samp)
    
    means = mean[:, etabin]
    means2fit = means[means!=0.0]
    ptbins2fit = ptbins[:-1][means!=0.0]
    meanvar2fit = np.abs(meanvar[means!=0.0,etabin])
    
    xvals = np.linspace(ptbins2fit.min() - (0.5), ptbins2fit.max()+(ptbins2fit[-1]-ptbins2fit[-11]),1000)
    
    import scipy as sp
    
    def ptscale2int(x, ptbins):
        ptmax = max(np.log10(ptbins))
        ptmin = min(np.log10(ptbins))
        z = np.log10(x)
        return ((z-ptmin)-(ptmax-z))/(ptmax-ptmin);
    
    def sum_cheb_tmp(x, ptbins, *p):
        c0, c1, c2, c3, c4 = p
        xs = ptscale2int(x, ptbins)
        res = (c0 * sp.special.eval_chebyt(0, xs) +
               c1 * sp.special.eval_chebyt(1, xs) + 
               c2 * sp.special.eval_chebyt(2, xs) + 
               c3 * sp.special.eval_chebyt(3, xs) + 
               c4 * sp.special.eval_chebyt(4, xs) )
        return res
    
    def sum_cheb5_tmp(x, ptbins, *p):
        c0, c1, c2, c3, c4, c5 = p
        xs = ptscale2int(x, ptbins)
        res = (c0 * sp.special.eval_chebyt(0, xs) +
               c1 * sp.special.eval_chebyt(1, xs) + 
               c2 * sp.special.eval_chebyt(2, xs) + 
               c3 * sp.special.eval_chebyt(3, xs) + 
               c4 * sp.special.eval_chebyt(4, xs) + 
               c5 * sp.special.eval_chebyt(5, xs) )
        return res
    
    def sum_cheb3_tmp(x, ptbins, *p):
        c0, c1, c2, c3 = p
        xs = ptscale2int(x, ptbins)
        res = (c0 * sp.special.eval_chebyt(0, xs) +
               c1 * sp.special.eval_chebyt(1, xs) + 
               c2 * sp.special.eval_chebyt(2, xs) + 
               c3 * sp.special.eval_chebyt(3, xs) )
        return res
    
    def sum_cheb2_tmp(x, ptbins, *p):
        c0, c1, c2 = p
        xs = ptscale2int(x, ptbins)
        res = (c0 * sp.special.eval_chebyt(0, xs) +
               c1 * sp.special.eval_chebyt(1, xs) + 
               c2 * sp.special.eval_chebyt(2, xs) )
        return res
    
    def response_fnc(x, *p):
        p0, p1, p2, p3, p4, p5 = p
        return p0+(p1/((np.log10(x)**2)+p2)) + (p3*np.exp(-p4*((np.log10(x)-p5)*(np.log10(x)-p5))))
    
    init_vals_2014 = {
        'b':
        [[0.540002, 13.8495, 17.8549, -0.215711, 0.576285, 1.42258],
        [0.73119, 7.52866, 17.3681, -0.078402, 1.21665, 1.69878],
        [0.999952, 0.0322738, -1.05606, -19.6994, 0.720321, -1.58314],
        [0.135913, 7.92441, 3.85698, -0.804604, 1.11911, 0.732041]],
        'c' :
        [[ 0.940259, 0.705481, 0.23917, -0.826926, 0.311473, -0.514041],
        [0.982083, 0.238007, 4.35924, -0.0314618, 5.91028, 1.67749],
        [0.733505, 7.26794, 12.2028, -0.756302, 0.0895257, -1.96324],
        [0.932305, 1.15954, 17.1731, -0.471313, 2.58424, 0.254917]],
        'g' :
        [[0.877892, 3.10194, 1.16568, -677.876, 0.0325026, -12.9485],
        [0.983775, 0.247943, 1.55373, -0.0254802, 3.35748, 1.71263],
        [-0.972548, 38.8683, 2.47151, -44.0233, 0.0901665, -3.15495],
        [1.0655, -0.0680325, -0.509038, -8.59434e+06, 42.6162, 0.357177]],
        'l':
        [[1.28488, -46.3648, 151.749, -0.0108461, 15.4256, 1.63377],
        [ 1.50931, -118.71, 224.19, -0.0196468, 4.62655, 1.51581],
        [0.692016, 8.26488, 11.1655, -0.802769, 0.116182, -1.16094],
        [1.01244, -0.0926519, -0.12138, -3.69494e+07, 7.15634, -0.625288]],   
    }
    
    init_vals_2014['b'][0] = [ 9.81014871e-01, -6.46744813e-03, -1.05658840e+00,  5.35445486e+03, 2.99200015e+01,  1.21399356e+02]
    init_vals_2014['b'][3] = [ 9.81014871e-01, -6.46744813e-03, -1.05658840e+00,  5.35445486e+03, 2.99200015e+01,  1.21399356e+02]
    
    def sum_cheb_tmp(x, ptbins, *p):
        c0, c1, c2, c3, c4 = p
        xs = ptscale2int(x, ptbins)
        res = (c0 * sp.special.eval_chebyt(0, xs) +
               c1 * sp.special.eval_chebyt(1, xs) + 
               c2 * sp.special.eval_chebyt(2, xs) + 
               c3 * sp.special.eval_chebyt(3, xs) + 
               c4 * sp.special.eval_chebyt(4, xs) )
        return res
    
    subsamples = ['_b', '_c', '_l', '_g']
    subsamples = ['_b', '_c']
    
    for samp in subsamples:
        for etabin in range(4):
            mean = read_data("Mean", samp)[:,etabin]
            meanvar = read_data("MeanVar", samp)[:,etabin]
    
            mean_range = range(15,len(mean)-8)
    
            means = mean[mean_range]
            ptbins2fit = ptbins_c[mean_range]
    
            means2fit = means[means!=0.0]
            ptbins2fit = ptbins2fit[means!=0.0]
            meanvar2fit = np.abs(meanvar[mean_range][means!=0.0])
    
            xvals = np.linspace(ptbins2fit.min() - (1), ptbins2fit.max()+50,1000) #-200 to plot the fitted curved outside of the points
            def sum_cheb(x, *p):
                return sum_cheb_tmp(x, ptbins2fit, *p)
            def sum_cheb5(x, *p):
                return sum_cheb5_tmp(x, ptbins2fit, *p)    
            def sum_cheb3(x, *p):
                return sum_cheb3_tmp(x, ptbins2fit, *p)  
            def sum_cheb2(x, *p):
                return sum_cheb2_tmp(x, ptbins2fit, *p)  
                
            try:
                p_resp, arr = curve_fit(response_fnc, ptbins2fit, means2fit, p0=init_vals_2014[samp[-1:]][etabin]) #0.793149, 4.60568, 11.1553, -0.123262, 0.878497, 1.52041
                p_resp_err, arr = curve_fit(response_fnc, ptbins2fit, means2fit, p0=p_resp, sigma=np.sqrt(meanvar2fit))
            except(RuntimeError):
                print("Winter 14 fit failed")
                p_resp, p_resp_err = [[np.nan]*6]*2
            #      = np.nan*6
    
            # p_resp_err
            p_cheb1, arr = curve_fit(sum_cheb, ptbins2fit, means2fit, p0=[ 1, 1, 1, 1, 1])
            p_cheb, arr = curve_fit(sum_cheb, ptbins2fit, means2fit, p0=p_cheb1, sigma=np.sqrt(meanvar2fit))
            p_cheb5_1, arr = curve_fit(sum_cheb5, ptbins2fit, means2fit, p0=[ 1, 1, 1, 1, 1, 1])
            p_cheb5, arr = curve_fit(sum_cheb5, ptbins2fit, means2fit, p0=p_cheb5_1, sigma=np.sqrt(meanvar2fit))
            p_cheb3_1, arr = curve_fit(sum_cheb3, ptbins2fit, means2fit, p0=[ 1, 1, 1, 1])
            p_cheb3, arr = curve_fit(sum_cheb3, ptbins2fit, means2fit, p0=p_cheb3_1, sigma=np.sqrt(meanvar2fit))
            p_cheb2_1, arr = curve_fit(sum_cheb2, ptbins2fit, means2fit, p0=[ 1, 1, 1])
            p_cheb2, arr = curve_fit(sum_cheb2, ptbins2fit, means2fit, p0=p_cheb2_1, sigma=np.sqrt(meanvar2fit))
    
            yvals_resp = response_fnc(xvals, *p_resp)
            yvals_resp_err = response_fnc(xvals, *p_resp_err)
            yvals = sum_cheb(xvals, *p_cheb)
            yvalsC5 = sum_cheb5(xvals, *p_cheb5)
            yvalsC3 = sum_cheb3(xvals, *p_cheb3)
            yvalsC2 = sum_cheb2(xvals, *p_cheb2)
    
            chi2_resp = np.sum((response_fnc(ptbins2fit, *p_resp_err) - means2fit)**2/meanvar2fit)
            chi2_C4 = np.sum((sum_cheb(ptbins2fit, *p_cheb) - means2fit)**2/meanvar2fit)
            chi2_C5 = np.sum((sum_cheb5(ptbins2fit, *p_cheb5) - means2fit)**2/np.abs(meanvar2fit))
            chi2_C3 = np.sum((sum_cheb3(ptbins2fit, *p_cheb3) - means2fit)**2/np.abs(meanvar2fit))
            chi2_C2 = np.sum((sum_cheb2(ptbins2fit, *p_cheb2) - means2fit)**2/np.abs(meanvar2fit))
            Ndof_C4 = len(ptbins2fit) - 5
            Ndof_C5 = len(ptbins2fit) - 6
            Ndof_C3 = len(ptbins2fit) - 4
            Ndof_C2 = len(ptbins2fit) - 3
            Ndof_resp = len(ptbins2fit) - 6
    
            fig, ax = plt.subplots()
    
            plt.errorbar(ptbins2fit, means2fit, yerr=np.sqrt(np.abs(meanvar2fit)), marker='o',
                         linestyle="none", label=f'Data {etabins_mod[etabin]}'+r'$<\eta<'+f'${etabins_mod[etabin+1]}')
    
            eta_string = '_eta'+str(etabins_mod[etabin])+'to'+str(etabins_mod[etabin+1])
            eta_string = eta_string.replace('.','')
            if np.isnan(chi2_resp): 
                winter14_lab = 'Winter14 func, failed'
            else:
                winter14_lab= 'Winter14 func, '+r'$\chi^2/N_{dof} = $'+r' {0:0.3g}/{1:0.0f}'.format(chi2_resp, Ndof_resp)
    
            ax.plot(xvals, yvals_resp_err, label=winter14_lab, linewidth=1.8);
            ax.plot(xvals, yvals, label=r'Chebyshev, n=4, '+r'$\chi^2/N_{dof} = $'+r' {0:0.3g}/{1:0.0f}'.format(chi2_C4, Ndof_C4),linewidth=1.8);
            ax.plot(xvals, yvalsC3, label=r'Chebyshev, n=3, '+r'$\chi^2/N_{dof} = $'+r' {0:0.3g}/{1:0.0f}'.format(chi2_C3, Ndof_C3),linewidth=1.8);
            ax.plot(xvals, yvalsC2, label=r'Chebyshev, n=2, '+r'$\chi^2/N_{dof} = $'+r' {0:0.3g}/{1:0.0f}'.format(chi2_C2, Ndof_C2),linewidth=1.8);
    
        
        #     print("ynorm = ", y_norm)
        #     print("yerr = ", yerr_norm)
            std = np.sqrt(np.abs(meanvar2fit))
            norm_pos = (std<0.1) &  (std != np.inf)
        #     print("(yerr_norm+y_norm)[norm_pos]) = ", (yerr_norm+y_norm)[norm_pos])
            ax.set_ylim(np.min((means2fit-std)[norm_pos])-0.002 ,np.max((std+means2fit)[norm_pos])+0.002)
    
            ax.set_xlabel(r'$p_T$ (GeV)');
            ax.set_ylabel(r'mean response');
            ax.set_xscale('log')
            # ax.set_ylim([0.8,1.1])
            ax.legend(prop={'size': 7})
            if not test_run:
                plt.savefig('fig/response_fit'+samp+eta_string+'.png', dpi=plt.rcParamsDefault['figure.dpi']);
                plt.savefig('fig/response_fit'+samp+eta_string+'.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
            else:
                plt.savefig('test/fig/response_fit'+samp+eta_string+'.png', dpi=plt.rcParamsDefault['figure.dpi']);
                plt.savefig('test/fig/response_fit'+samp+eta_string+'.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
            # fig.set_size_inches(6, 4, forward=True)
            plt.show();
            plt.close();
    'a';
    
    def read_data(name, samp, tag_full):
        df_csv = pd.read_csv('out_txt/EtaBinsvsPtBins'+name+samp+tag_full+'.csv').set_index('etaBins')
        
        data = df_csv.to_numpy().transpose()
        return data
    
    def make_comparison_plot(yvar1, std1, yvar2, std2, k, samp, name1:str, name2:str):
        ''' Make a coparison and a ratio plot of yvar2 vs yvar1
        '''
        
        mean_p_QCD = yvar1.copy()
        mean_p_QCD[mean_p_QCD==0] = np.nan
    
        mean_p_TTBAR = yvar2.copy()
        mean_p_TTBAR[mean_p_TTBAR==0] = np.nan
    
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        start = 17
    
        for axis in [ax.xaxis, ax.yaxis, ax2.xaxis, ax2.yaxis]:
            axis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    
        xvals = (ptbins[start:-1] + ptbins[start+1:])/2
        wd = np.abs(ptbins[start:-1] - ptbins[start+1:])
        QCD_vals = mean_p_QCD[start:,k]
        TTbar_vals = mean_p_TTBAR[start:,k]
    
        p1 = ax.errorbar(xvals, QCD_vals, yerr=std1[start:,k], marker='o',
                     linestyle="none", label='QCD')
    
        p5 = ax.errorbar(xvals, TTbar_vals, yerr=std2[start:,k], marker='v',
                     linestyle="none", label='TTBAR')
    
        ax2.hlines(1,1, 10000, linestyles='--',color="black",
            linewidth=1,)
        rel_mc_unc =  std1[start:,k]/QCD_vals 
    
        ax2.bar(
            xvals,
            2 * rel_mc_unc,
            width=wd,
            bottom=1.0 - rel_mc_unc,
            fill=False,
            linewidth=0,
            edgecolor="gray",
            hatch=10 * "/",
        )
    
        # data in ratio plot
        data_model_ratio = TTbar_vals/QCD_vals
        data_model_ratio_unc = std2[start:,k] / QCD_vals
        # mask data in bins where total model yield is 0
        ax2.errorbar(
            xvals,
            data_model_ratio, #[nonzero_model_yield],
            yerr=data_model_ratio_unc, #[nonzero_model_yield],
            fmt="o",
            color="k",
        )
    
        ax2.set_ylabel(name2+" / "+name1)
        ax2.tick_params(axis="both", which="major", pad=8)
        ax2.tick_params(direction="in", top=True, right=True, which="both")
    
        fig.set_tight_layout(True)
    
        ### Calculate resonable limits excluding the few points with insane errors
        yerr_norm = np.concatenate([std1[start:,k], std1[start:,k] ])
        y_norm = np.concatenate([QCD_vals, TTbar_vals])
        norm_pos = (yerr_norm<0.03) &  (yerr_norm != np.inf) & (y_norm>-0.1)
        ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]), np.max((yerr_norm+y_norm)[norm_pos]))
    
        yerr_norm = np.concatenate([std1[start:,k]/QCD_vals, std2[start:,k]/QCD_vals ])
        y_norm = np.concatenate([QCD_vals/QCD_vals, TTbar_vals/QCD_vals])
        norm_pos = (yerr_norm<0.03) &  (yerr_norm != np.inf) & (y_norm>-0.1) 
        ax2.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]), np.max((yerr_norm+y_norm)[norm_pos]))
    
        ax2.set_xlabel(r'$p_T$ (GeV)');
        ax.set_ylabel(r'mean response');
        ax.set_xscale('log')
        ax2.set_xscale('log')
    
        ax.set_xticks([])
        ax2.set_xticks([20, 50, 100, 500, 1000, 5000])
        ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        leg1 = ax.legend()
    
        ax2.set_xlim(ax.get_xlim())
        eta_string = '_eta'+str(etabins_mod[k])+'to'+str(etabins_mod[k+1])
        eta_string = eta_string.replace('.','')
    
        plt.savefig('fig/corr_vs_pt'+samp+eta_string+'_L5_'+name1+'-'+name2+'-median.pdf');
        plt.savefig('fig/corr_vs_pt'+samp+eta_string+'_L5_'+name1+'-'+name2+'-median.png');
        # gs1.tight_layout(fig, rect=[0, 0.1, 0.8, 0.5])
        plt.show();
    
    subsamples = ['', '_b', '_c', '_l', '_g']
    
    k2 = np.where(etabins_mod<=0)[0][-1]
    k4 = np.where(etabins_mod<=1.3)[0][-1]
    k6 = np.where(etabins_mod<=2.5)[0][-1]
    k8 = np.where(etabins_mod<=3.0)[0][-1]
    ks = [k2, k4, k6, k8]
    
    for samp in subsamples:
        for k in ks:
            print('Fitting subsample: ', samp)
            median_QCD = read_data("Median", samp, '_L5_QCD')
            medianstd_QCD = read_data("MedianStd", samp, '_L5_QCD')
            median_TTBAR = read_data("Median", samp, '_L5_TTBAR')
            medianstd_TTBAR = read_data("MedianStd", samp, '_L5_TTBAR')
            
            
            make_comparison_plot(median_QCD, medianstd_QCD, median_TTBAR, medianstd_TTBAR,
                                 k, samp, 'QCD', 'TTBAR')
            
    3;
    
    subsamples = ['', '_b', '_c', '_l', '_g']
    
    k2 = np.where(etabins_mod<=0)[0][-1]
    k4 = np.where(etabins_mod<=1.3)[0][-1]
    k6 = np.where(etabins_mod<=2.5)[0][-1]
    k8 = np.where(etabins_mod<=3.0)[0][-1]
    ks = [k2, k4, k6, k8]
    
    ks = [k2, k4] #, k6, k8]
    subsamples = [''] #, '_b']
    for samp in subsamples:
        for k in ks:
            print('Fitting subsample: ', samp)
            mean_QCD = read_data("Mean", samp, '_L5_QCD')
            meanvar_QCD = read_data("MeanVar", samp, '_L5_QCD')
            mean_TTBAR = read_data("Mean", samp, '_L5_TTBAR')
            meanvar_TTBAR = read_data("MeanVar", samp, '_L5_TTBAR')
            
            
            make_comparison_plot(mean_QCD, np.sqrt(np.abs(meanvar_QCD)), mean_TTBAR, np.sqrt(np.abs(meanvar_TTBAR)),
                                 k, samp, 'QCD', 'TTBAR')
            
    3;
    
    subsamples = ['', '_b', '_c', '_l', '_g']
    
    k2 = np.where(etabins_mod<=0)[0][-1]
    k4 = np.where(etabins_mod<=1.3)[0][-1]
    k6 = np.where(etabins_mod<=2.5)[0][-1]
    k8 = np.where(etabins_mod<=3.0)[0][-1]
    ks = [k2, k4, k6, k8]
    
    ks = [k2, k4] #, k6, k8]
    subsamples = [''] #, '_b']
    for samp in subsamples:
        for k in ks:
            print('Fitting subsample: ', samp)
            mean_QCD = read_data("Mean", samp, '_L5_QCD')
            meanvar_QCD = read_data("MeanVar", samp, '_L5_QCD')
            mean_TTBAR = read_data("Mean", samp, '_L5_TTBAR')
            meanvar_TTBAR = read_data("MeanVar", samp, '_L5_TTBAR')
    
            mean_p_QCD = mean_QCD.copy()
            mean_p_QCD[mean_p_QCD==0] = np.nan
    
            mean_p_TTBAR = mean_TTBAR.copy()
            mean_p_TTBAR[mean_p_TTBAR==0] = np.nan
    
            gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            start = 17
    
            p1 = plt.errorbar(ptbins[start:-1],mean_p_QCD[start:,k], yerr=np.sqrt(meanvar_QCD[start:,k]), marker='o',
                         markerfacecolor='none', markeredgewidth=1,
                         linestyle="none", label=f'${etabins_mod[k]}<\eta<{etabins_mod[k+1]}$')
    
            p5 = plt.errorbar(ptbins[start:-1],mean_p_TTBAR[start:,k], yerr=np.sqrt(meanvar_TTBAR[start:,k]), marker='v',
                         markerfacecolor='none', markeredgewidth=1, color=p1.lines[0].get_color(),
                         linestyle="none")
    
            ax3 = ax.twinx()
            ax3.plot(np.nan, marker='o',color='k',markerfacecolor='none', markeredgewidth=1, linestyle="none", label='QCD')
            ax3.plot(np.nan, marker='v',color='k',markerfacecolor='none', markeredgewidth=1, linestyle="none", label='TTBAR')
    
            ### Calculate resonable limits excluding the few points with insane errors
            yerr_norm = np.concatenate([np.sqrt(meanvar[start:,[k2, k4, k6, k8]]) ])
            y_norm = np.concatenate([mean_p_TTBAR[start:,[k2, k4, k6, k8]]])
            norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
            ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]) ,np.max((yerr_norm+y_norm)[norm_pos]))
    
            ax.set_xlabel(r'$p_T$ (GeV)');
            ax.set_ylabel(r'mean response');
            ax.set_xscale('log')
            ax.set_xticks([20, 50,100, 500, 1000, 5000])
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            leg1 = ax.legend()
            leg1 = ax3.legend(loc='upper left')
            eta_string = '_eta'+str(etabins_mod[k])+'to'+str(etabins_mod[k+1])
            eta_string = eta_string.replace('.','')
            
            plt.savefig('fig/corr_vs_pt'+samp+eta_string+'_L5_QCD_TTBAR.pdf');
            plt.savefig('fig/corr_vs_pt'+samp+eta_string+'_L5_QCD_TTBAR.png');
            # gs1.tight_layout(fig, rect=[0, 0.1, 0.8, 0.5])
            plt.show();
    3;
    
    subsamples = ['', '_b', '_c', '_l', '_g']
    
    k2 = np.where(etabins_mod<=0)[0][-1]
    k4 = np.where(etabins_mod<=1.3)[0][-1]
    k6 = np.where(etabins_mod<=2.5)[0][-1]
    k8 = np.where(etabins_mod<=3.0)[0][-1]
    ks = [k2, k4, k6, k8]
    
    for samp in subsamples:
        for k in ks:
            print('Fitting subsample: ', samp)
            mean_QCD = read_data("Mean", samp, '_L5_QCD')
            meanvar_QCD = read_data("MeanVar", samp, '_L5_QCD')
            mean_TTBAR = read_data("Mean", samp, '_L5_TTBAR')
            meanvar_TTBAR = read_data("MeanVar", samp, '_L5_TTBAR')
            
            
            mean_p_QCD = mean_QCD.copy()
            mean_p_QCD[mean_p_QCD==0] = np.nan
    
            mean_p_TTBAR = mean_TTBAR.copy()
            mean_p_TTBAR[mean_p_TTBAR==0] = np.nan
    
            fig = plt.figure()
            gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
            ax = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            start = 17
        
        
            for axis in [ax.xaxis, ax.yaxis, ax2.xaxis, ax2.yaxis]:
                axis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    
            xvals = (ptbins[start:-1] + ptbins[start+1:])/2
            wd = np.abs(ptbins[start:-1] - ptbins[start+1:])
            QCD_vals = mean_p_QCD[start:,k]
            TTbar_vals = mean_p_TTBAR[start:,k]
    
            p1 = ax.errorbar(xvals, QCD_vals, yerr=np.sqrt(meanvar_QCD[start:,k]), marker='o',
                         linestyle="none", label='QCD')
    
            p5 = ax.errorbar(xvals, TTbar_vals, yerr=np.sqrt(meanvar_TTBAR[start:,k]), marker='v',
                         linestyle="none", label='TTBAR')
            
            ax2.hlines(1,1, 10000, linestyles='--',color="black",
                linewidth=1,)
            # ratio plot
            
            rel_mc_unc =  np.sqrt(meanvar_QCD[start:,k])/QCD_vals 
            
            ax2.bar(
                xvals,
                2 * rel_mc_unc,
                width=wd,
                bottom=1.0 - rel_mc_unc,
                fill=False,
                linewidth=0,
                edgecolor="gray",
                hatch=10 * "/",
            )
            
            
    
            # data in ratio plot
            data_model_ratio = TTbar_vals/QCD_vals
            data_model_ratio_unc = np.sqrt(meanvar_TTBAR[start:,k]) / QCD_vals
            # mask data in bins where total model yield is 0
            ax2.errorbar(
                xvals,
                data_model_ratio, #[nonzero_model_yield],
                yerr=data_model_ratio_unc, #[nonzero_model_yield],
                fmt="o",
                color="k",
            )
            
            
    
            
            ax2.set_ylabel("TTbar / QCD")
            ax2.tick_params(axis="both", which="major", pad=8)
            ax2.tick_params(direction="in", top=True, right=True, which="both")
    
            fig.set_tight_layout(True)
    
            ### Calculate resonable limits excluding the few points with insane errors
            yerr_norm = np.concatenate([np.sqrt(meanvar_QCD[start:,k]), np.sqrt(meanvar_TTBAR[start:,k]) ])
            y_norm = np.concatenate([QCD_vals, TTbar_vals])
            norm_pos = (yerr_norm<0.03) &  (yerr_norm != np.inf) & (y_norm>-0.1)
            ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]), np.max((yerr_norm+y_norm)[norm_pos]))
            
            yerr_norm = np.concatenate([np.sqrt(meanvar_QCD[start:,k])/QCD_vals, np.sqrt(meanvar_TTBAR[start:,k])/QCD_vals ])
            y_norm = np.concatenate([QCD_vals/QCD_vals, TTbar_vals/QCD_vals])
            norm_pos = (yerr_norm<0.03) &  (yerr_norm != np.inf) & (y_norm>-0.1) 
            ax2.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]), np.max((yerr_norm+y_norm)[norm_pos]))
    
            ax2.set_xlabel(r'$p_T$ (GeV)');
            ax.set_ylabel(r'mean response');
            ax.set_xscale('log')
            ax2.set_xscale('log')
            
            ax.set_xticks([])
            ax2.set_xticks([20, 50,100, 500, 1000, 5000])
            ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            leg1 = ax.legend()
            # fig.legend((p1, p5), ('QCD', 'TTBAR'), loc='upper right')  #,bbox_to_anchor=(0.5, 0., 0.47, 0.98)) #loc="upper right")
            # if test_run:
            #     plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
            #     plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.png', dpi=plt.rcParamsDefault['figure.dpi']);
            # else:
            ax2.set_xlim(ax.get_xlim())
            eta_string = '_eta'+str(etabins_mod[k])+'to'+str(etabins_mod[k+1])
            eta_string = eta_string.replace('.','')
            
            plt.savefig('fig/corr_vs_pt'+samp+eta_string+'_L5_QCD_TTBAR.pdf');
            plt.savefig('fig/corr_vs_pt'+samp+eta_string+'_L5_QCD_TTBAR.png');
            # gs1.tight_layout(fig, rect=[0, 0.1, 0.8, 0.5])
            plt.show();
    3;
    
    subsamples = ['', '_b', '_c']
    
    k2 = np.where(etabins_mod<=0)[0][-1]
    k4 = np.where(etabins_mod<=1.3)[0][-1]
    k6 = np.where(etabins_mod<=2.5)[0][-1]
    k8 = np.where(etabins_mod<=3.0)[0][-1]
    ks = [k2, k4, k6, k8]
    
    ks = [k2, k4, k6, k8] #, k6, k8]
    for samp in subsamples:
        for k in ks:
            print('Fitting subsample: ', samp)
            mean_TTBAR = read_data("Mean", samp, '_L5_TTBAR-hadflav-0-50')
            meanvar_TTBAR = read_data("MeanVar", samp, '_L5_TTBAR-hadflav-0-50')
            mean_Herwig = read_data("Mean", samp, '_L5_Herwig-TTBAR')
            meanvar_Herwig = read_data("MeanVar", samp, '_L5_Herwig-TTBAR')
    
            mean_p_Herwig = mean_Herwig.copy()
            mean_p_Herwig[mean_p_Herwig==0] = np.nan
    
            mean_p_TTBAR = mean_TTBAR.copy()
            mean_p_TTBAR[mean_p_TTBAR==0] = np.nan
    
            fig = plt.figure()
            gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
            ax = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            start = 17
        
    
            for axis in [ax.xaxis, ax.yaxis, ax2.xaxis, ax2.yaxis]:
                axis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    
            xvals = (ptbins[start:-1] + ptbins[start+1:])/2
            wd = np.abs(ptbins[start:-1] - ptbins[start+1:])
            Herwig_vals = mean_p_Herwig[start:,k]
            TTbar_vals = mean_p_TTBAR[start:,k]
    
            p5 = ax.errorbar(xvals, TTbar_vals, yerr=np.sqrt(meanvar_TTBAR[start:,k]), marker='o',
                         linestyle="none", label='Pythia')
          
            p1 = ax.errorbar(xvals, Herwig_vals, yerr=np.sqrt(meanvar_Herwig[start:,k]), marker='v',
                         linestyle="none", label='Herwig')
    
       
            ax2.hlines(1,1, 10000, linestyles='--',color="black",
                linewidth=1,)
            rel_mc_unc =  np.sqrt(meanvar_TTBAR[start:,k])/TTbar_vals 
            
            ax2.bar(
                xvals,
                2 * rel_mc_unc,
                width=wd,
                bottom=1.0 - rel_mc_unc,
                fill=False,
                linewidth=0,
                edgecolor="gray",
                hatch=10 * "/",
            )
            
            data_model_ratio = Herwig_vals/TTbar_vals
            data_model_ratio_unc = np.sqrt(meanvar_Herwig[start:,k]) / TTbar_vals
            # mask data in bins where total model yield is 0
            ax2.errorbar(
                xvals,
                data_model_ratio, #[nonzero_model_yield],
                yerr=data_model_ratio_unc, #[nonzero_model_yield],
                fmt="o",
                color="k",
            )
            
            
            ax2.set_ylabel("Herwig / Pythia")
            ax2.tick_params(axis="both", which="major", pad=8)
            ax2.tick_params(direction="in", top=True, right=True, which="both")
    
            fig.set_tight_layout(True)
    
            ### Calculate resonable limits excluding the few points with insane errors
            yerr_norm = np.concatenate([np.sqrt(meanvar_Herwig[start:,k]), np.sqrt(meanvar_TTBAR[start:,k]) ])
            y_norm = np.concatenate([Herwig_vals, TTbar_vals])
            norm_pos = (yerr_norm<0.03) &  (yerr_norm != np.inf) & (y_norm>-0.1)
            ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]), np.max((yerr_norm+y_norm)[norm_pos]))
            
            yerr_norm = np.concatenate([np.sqrt(meanvar_Herwig[start:,k])/TTbar_vals, np.sqrt(meanvar_TTBAR[start:,k])/TTbar_vals ])
            y_norm = np.concatenate([Herwig_vals/TTbar_vals, TTbar_vals/TTbar_vals])
            norm_pos = (yerr_norm<0.03) &  (yerr_norm != np.inf) & (y_norm>-0.1) 
            ax2.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]), np.max((yerr_norm+y_norm)[norm_pos]))
    
            ax2.set_xlabel(r'$p_T$ (GeV)');
            ax.set_ylabel(r'mean response');
            ax.set_xscale('log')
            ax2.set_xscale('log')
            
            ax.set_xticks([])
            ax2.set_xticks([20, 50,100, 500, 1000, 5000])
            ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            leg1 = ax.legend()
    
            ax2.set_xlim(ax.get_xlim())
            eta_string = '_eta'+str(etabins_mod[k])+'to'+str(etabins_mod[k+1])
            eta_string = eta_string.replace('.','')
            
            plt.savefig('fig/corr_vs_pt'+samp+eta_string+'_L5_Herwig_Pythia-hadflav.pdf');
            plt.savefig('fig/corr_vs_pt'+samp+eta_string+'_L5_Herwig_Pythia-hadflav.png');
            # gs1.tight_layout(fig, rect=[0, 0.1, 0.8, 0.5])
            plt.show();
    3;
    
    yerr_norm = np.concatenate([np.sqrt(meanvar_QCD[start:,k]), np.sqrt(meanvar_TTBAR[start:,k]) ])
    y_norm = np.concatenate([QCD_vals, TTbar_vals])
    norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
    
    def plot_corrections(mean, samp, meanvar):
        ### To ignore the points with 0 on y axis when setting the y axis limits
        mean_p = mean.copy()
        mean_p[mean_p==0] = np.nan
    
        # h = np.max(histo.values()[('QCD',)])
        fig, ax = plt.subplots()
        start = 17
        
        
        k2 = np.where(etabins_mod<=0)[0][-1]
        k4 = np.where(etabins_mod<=1.3)[0][-1]
        k6 = np.where(etabins_mod<=2.5)[0][-1]
        k8 = np.where(etabins_mod<=3.0)[0][-1]
        
    
        plt.errorbar(ptbins[start:-1],mean_p[start:,k2], yerr=np.sqrt(meanvar[start:,k2]), marker='o',
                     linestyle="none", label=f'${etabins_mod[k2]}<\eta<{etabins_mod[k2+1]}$')
        plt.errorbar(ptbins[start:-1],mean_p[start:,k4], yerr=np.sqrt(meanvar[start:,k4]), marker='o',
                 linestyle="none", label=f'${etabins_mod[k4]}<\eta<{etabins_mod[k4+1]}$')
        plt.errorbar(ptbins[start:-1],mean_p[start:,k6], yerr=np.sqrt(meanvar[start:,k6]), marker='o',
                 linestyle="none", label=f'${etabins_mod[k6]}<\eta<{etabins_mod[k6+1]}$')
        plt.errorbar(ptbins[start:-1],mean_p[start:,k8], yerr=np.sqrt(meanvar[start:,k8]), marker='o',
                 linestyle="none", label=f'${etabins_mod[k8]}<\eta<{etabins_mod[k8+1]}$')
    
        ### Calculate resonable limits excluding the few points with insane errors
        yerr_norm = np.concatenate([np.sqrt(meanvar[start:,[k2, k4, k6, k8]]) ])
        y_norm = np.concatenate([mean_p[start:,[k2, k4, k6, k8]]])
        norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
        ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]) ,np.max((yerr_norm+y_norm)[norm_pos]))
    
        ax.set_xlabel(r'$p_T$ (GeV)');
        ax.set_ylabel(r'mean response');
        ax.set_xscale('log')
        # ax.set_ylim([0.8,1.1])
        ax.legend()
        if test_run:
            plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
            plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.png', dpi=plt.rcParamsDefault['figure.dpi']);
        else:
            plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.pdf');
            plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.png');
    
        plt.show();
        
    
    
    
    print('-----'*10)
    print("All done. Congrats!")
    
if __name__ == "__main__":
    main()