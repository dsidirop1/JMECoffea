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
    CERNCondorCluster = True
    CoffeaCasaEnv     = False
    load_preexisting  = False    ### True if don't repeat the processing of files and use preexisting JER from output
    test_run          = False     ### True if run only on one file
    load_fit_res      = False
    
    fine_etabins      = False
    one_bin           = False
    
    tag = '_L5'
    
    exec('from CoffeaJERCProcessor'+tag+' import Processor') 
    
    add_tag = '_LHEflav1_TTBAR-JME-condortest' #Herwig-TTBAR-JME-noIso' #'_Herwig-TTBAR' # '_TTBAR' #'_QCD' # '_testing_19UL18' # '' #fine_etaBins
    
    tag_full = tag+add_tag
    
    outname = 'out/CoffeaJERCOutputs'+tag_full+'.coffea'
    outname = outname+'_test' if test_run else outname
    
    if one_bin:
        tag_full='_oneBin'+tag_full
        
    if load_preexisting == True:
        UsingDaskExecutor = False
        
    if UsingDaskExecutor == False:
        CERNCondorCluster = False
        
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
        
    dataset = 'fileNames/fileNames_TTToSemi20UL18.txt'
    dataset = 'fileNames/fileNames_QCD20UL18.txt'
    dataset = 'fileNames/fileNames_QCD20UL18_JMENano.txt'
    dataset = 'fileNames/fileNames_TTToSemi20UL18_JMENano.txt'
#     dataset = 'fileNames/fileNames_Herwig_20UL18_JMENano.txt'
    
    
    rootfiles = open(dataset).read().split()
    
    fileslist = [xrootdstr + file for file in rootfiles]
    fileslist = fileslist[:1] # if add_tag=='QCD' else fileslist # :20 to skim the events
    
    def find_xsec(key):
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
    
    xsec = find_xsec(dataset)
    
    datasets = ['fileNames/fileNames_TTToSemi20UL18_JMENano.txt',
               'fileNames/fileNames_TTToDilep20UL18_JMENano.txt',
               'fileNames/fileNames_TTToHad20UL18_JMENano.txt'
               ]
    
    for data_s in datasets:
        rootfiles = open(dataset).read().split()
        fileslist = [xrootdstr + file for file in rootfiles]
        fileslist = fileslist[:40]
    filesets = {'QCD': {"files": fileslist, "metadata": {"xsec": xsec}}}
    
    fileslist = fileslist
    if test_run:
        fileslist = [fileslist[1]]
    
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
    
    if(UsingDaskExecutor and CoffeaCasaEnv):
        client = Client("tls://ac-2emalik-2ewilliams-40cern-2ech.dask.coffea.casa:8786")
        client.upload_file('CoffeaJERCProcessor.py')
    
    if(UsingDaskExecutor and not CoffeaCasaEnv):
        from dask.distributed import Client 
        if not CERNCondorCluster:
            client = Client()
    
        else:
            from dask_lxplus import CernCluster
            import socket
    
            cluster = CernCluster(
                cores = 1,
                memory = '1200MB',
                disk = '20MB',
                death_timeout = '60',
                lcg = True,
                nanny = False,
                container_runtime = 'none',
                log_directory = '/eos/user/a/anpotreb/condor/log',
                scheduler_options = {
                    'port': 8786,
                    'host': socket.gethostname(),
                },
                job_extra = {
                    'MY.JobFlavour': '"workday"', #'"longlunch"',
        #             'transfer_input_files': '/afs/cern.ch/user/a/anpotreb/top/JERC/JMECoffea/CoffeaJERCProcessor_L5.py',
                },
            )
            cluster.adapt(minimum=2, maximum=50)
            cluster.scale(8)
            client = Client(cluster)
        
        client.upload_file('CoffeaJERCProcessor'+tag+'.py')
    
        client
    
    tstart = time.time()
    
    outputs_unweighted = {}
    
    seed = 1234577890
    prng = RandomState(seed)
    Chunk = [10000, 5] # [chunksize, maxchunks]
    
    filesets = {'QCD': {"files": fileslist, "metadata": {"xsec": xsec}}}
    
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
        print("Processor finished. Time elapsed: ", elapsed)
        outputs_unweighted[name] = output
        print("Saving the output histograms under: ", outname)
        util.save(output, outname)
        outputs_unweighted[name] = output
    else:
        output = util.load(outname)
        print("Loaded histograms from: ", outname)
       
    
    if UsingDaskExecutor:
        client.close()
        if CERNCondorCluster or CoffeaCasaEnv:
            cluster.close()
    
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
    f_xvals = np.linspace(0,5,5001)
    
    if fine_etabins==True:
        ptbins = np.array([15, 40, 150, 400, 4000, 10000])
        ptbins_c = (ptbins[:-1]+ptbins[1:])/2
        etabins = output['ptresponse'].axis('jeteta').edges()
    elif one_bin==True:
        ptbins = np.array([15, 10000])
        ptbins_c = (ptbins[:-1]+ptbins[1:])/2
        etabins = np.array([etabins[0], 0, etabins[-1]])
    else:
        ptbins = output['ptresponse'].axis('pt').edges()
        ptbins_c = output['ptresponse'].axis('pt').centers()
        etabins = np.array([-5, -3, -2.5, -1.3, 0, 1.3, 2.5, 3, 5])
    
        
    jetpt_length = len(ptbins)-1
    jeteta_length = (len(etabins)-1)//2
    
    etabins_mod = etabins[(len(etabins)-1)//2:]
    etabins_c = (etabins_mod[:-1]+etabins_mod[1:])/2 #output['ptresponse'].axis('jeteta').centers()
    
    ptresp_edd = output['ptresponse'].axis('ptresponse').edges()
    plot_pt_edges = ptresp_edd[0:np.nonzero(ptresp_edd>=2.0)[0][0]]
    hist_pt_edges = plot_pt_edges[1:-1]   #for plotting. To exclude overflow from the plot
    plot_response_axis = hist.Bin("jeteta", r"Jet $\eta$", hist_pt_edges)
    
    from pltStyle import pltStyle
    pltStyle(style='paper')
    plt.rcParams['figure.subplot.left'] = 0.162
    plt.rcParams['figure.dpi'] = 150
    
                
    
        
    
        
    
    combineTTbar = False
    if combineTTbar==True:
        N_evts_had = 41113000
        N_evts_semilep = 48065000
        N_evts_dilep = 41160000
    
        N_evts = np.array([N_evts_had, N_evts_semilep, N_evts_dilep])
    
        N_av = (N_evts_had+N_evts_semilep+N_evts_dilep)/3
    
        semilepxsec = 0.108*3*0.337*2*2
        dilepxsec = 0.108*3*0.108*3
        hadxsec = 0.337*2*0.337*2
    
        xsec = np.array([hadxsec, semilepxsec, dilepxsec])
        weights = N_evts/N_av*xsec
    
        # (N_evts/N_av*xsec).sum()
    
        add_tags = ['_LHEflav1_TTBAR-Had-JME', '_LHEflav1_TTBAR-JME', '_LHEflav1_TTBAR-Dilep-JME']
        tag_fulls = [tag+addtag for addtag in add_tags]
        outnames = ['out/CoffeaJERCOutputs'+tag_full+'.coffea' for tag_full in tag_fulls]
    
        outputs = [util.load(outname) for outname in outnames]
        
        tag_full = tag + '_LHEflav1_TTBAR-Inclusive-JME'
    
        
        
    
            
    
                
            
    
    combine_antiflavour = False
    
    import warnings
    barable_samples = ['_b', '_c', '_s', 'ud_']
    
    def fit_responses(output, samp='_b'):
        warnings.filterwarnings('ignore')
        
            
        
        if combineTTbar:
            if combine_antiflavour and (samp in barable_samples):
                response_hist = outputs[0]['ptresponse'+samp].copy() + outputs[0]['ptresponse'+samp+'bar'].copy()            
            else:
                response_hist = outputs[0]['ptresponse'+samp].copy()
            response_hist.scale(weights[0])
            for out, wg in zip(outputs[1:], weights[1:]):
                if combine_antiflavour and (samp in barable_samples):
                    response_hist2 = out['ptresponse'+samp].copy() + out['ptresponse'+samp+'bar'].copy()
                else:
                    response_hist2 = out['ptresponse'+samp].copy()
                response_hist2.scale(wg)
                response_hist.add(histo2)
        else:
            if combine_antiflavour and (samp in barable_samples):
                response_hist = output['ptresponse'+samp] + output['ptresponse'+samp+'bar']
            else:
                response_hist = output['ptresponse'+samp]
    
        
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
            
        xvals = response_hist.axis('ptresponse').centers()[1:] #[1:] to exclude the second peak for low pt
        f_xvals = np.linspace(0,max(xvals),5001)
        response_edges = response_hist.axis('ptresponse').edges()[1:]
    
        for i in range(jetpt_length):
            ptBin = hist.Interval(ptbins[i], ptbins[i+1])
            
            if not 'inf' in str(ptBin):
                pt_string = '_pT'+str(int(ptBin.lo))+'to'+str(int(ptBin.hi))
            else:
                pt_string = '_pT'+str(ptBin.lo) + 'to' + str(ptBin.hi)
                pt_string = pt_string.replace('.0','').replace('-infto','0to')
    
            for k in range(jeteta_length):
                etaBinPl = hist.Interval(etabins[k+jeteta_length], etabins[k+1+jeteta_length])
                etaBinMi = hist.Interval(etabins[jeteta_length-k-1], etabins[jeteta_length-k])
                eta_string = '_eta'+str(etaBinPl.lo)+'to'+str(etaBinPl.hi)
                eta_string = eta_string.replace('.','')
                
                # The name integrate is a bit misleasding in this line. Is there another way to "slice" a histogram? //Andris
                histoMi = response_hist.integrate('jeteta', etaBinMi).integrate('pt', ptBin)
                histoPl = response_hist.integrate('jeteta', etaBinPl).integrate('pt', ptBin)
                histo = (histoMi+histoPl)
                    
                yvals = histo.values()[('QCD',)][1:]  #[1:] to exclude the second peak for low pt
    
                N = histo.integrate('ptresponse').values()[('QCD',)]-histo.values()[('QCD',)][0]
                
               ####################### Calculate median and rms ############################
                
                yvals_cumsum = np.cumsum(yvals)
                   # For N<200 too little statistics to calculate the error resonably
                med_bin = np.nonzero(yvals_cumsum>N/2)[0][0] if N>200 else 0
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
                else:
                    try:
                        p, arr = curve_fit(gauss, xvals, yvals, p0=[10,1,1])
                        N_converge += 1
                             ######## Second Gaussian ########
                        xfit_l = np.where(xvals>=p[1]-np.abs(p[2])*1.5)[0][0]
                        xfit_hs = np.where(xvals>=p[1]+np.abs(p[2])*1.5)[0]
                        xfit_h = xfit_hs[0] if len(xfit_hs)>0 else len(xvals)
                        
                        if len(range(xfit_l,xfit_h))<6: #if there are only 3pnts, the uncertainty is infty
                            xfit_l = xfit_l-1
                            xfit_h = xfit_h+1
                            if len(range(xfit_l,xfit_h))<6:
                                xfit_l = xfit_l-1
                                xfit_h = xfit_h+1
                        if xfit_l<0:
                            xfit_h-=xfit_l
                            xfit_l = 0
                        xvals2 = xvals[xfit_l: xfit_h]
                        yvals2 = yvals[xfit_l: xfit_h]
                        p2, arr = curve_fit(gauss, xvals2, yvals2, p0=p)
    
                        ygaus = gauss(xvals, *p2)
                        chi2 = sum((yvals-ygaus)**2/(yvals+1E-9))
                        Ndof = len(xvals2)-3
                    except(RuntimeError):
                        p2=[0,0,0]
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
    
            if not test_run and (not fine_etabins) and True:
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
    
                    plt.close();                
    
        print("N converge = ", N_converge, "N_not_converge = ", N_not_converge );
        warnings.filterwarnings('default')
        
        return [mean, meanvar, medians, medianstds] #width, 
        
    
    def plot_corrections(mean, samp, meanstd):
        ### To ignore the points with 0 on y axis when setting the y axis limits
        mean_p = mean.copy()
        mean_p[mean_p==0] = np.nan
    
        fig, ax = plt.subplots()
        start = np.searchsorted(ptbins_c, 20, side='left') #np.where(ptbins<=20)[0][-1]
        
        ptbins_plot = ptbins_c[start:]
        meanstd = meanstd[start:,:]
        
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
        
        ax.set_xscale('log')
        ax.set_xticks([20, 50, 100, 500, 1000, 5000])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel(r'$p_T$ (GeV)');
        ax.set_ylabel(r'mean response');
        ax.legend()
        if test_run:
            plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
            plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.png', dpi=plt.rcParamsDefault['figure.dpi']);
        else:
            plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.pdf');
            plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.png');
    
        plt.show();
        
    
    def plot_corrections_eta(mean, samp, meanstd):
        ### To ignore the points with 0 on y axis when setting the y axis limits
        mean_p = mean.copy()
        mean_p[mean_p==0] = np.nan
    
        fig, ax = plt.subplots()
        
        
        k2 = np.where(ptbins<=15)[0][-1]
        k4 = np.where(ptbins<=40)[0][-1]
        k6 = np.where(ptbins<=150)[0][-1]
        k8 = np.where(ptbins<=400)[0][-1]
        
        plt.errorbar(etabins_c, mean_p[k2,:], yerr=meanstd[k2], marker='o',
                     linestyle="none", label=f'{ptbins[k2]}'+r'$<p_t<$'+f'{ptbins[k2+1]}')
        plt.errorbar(etabins_c, mean_p[k4,:], yerr=meanstd[k4], marker='o',
                 linestyle="none", label=f'{ptbins[k4]}'+r'$<p_t<$'+f'{ptbins[k4+1]}')
        plt.errorbar(etabins_c, mean_p[k6], yerr=meanstd[k6], marker='o',
                 linestyle="none", label=f'{ptbins[k6]}'+r'$<p_t<$'+f'{ptbins[k6+1]}')
        plt.errorbar(etabins_c, mean_p[k8], yerr=meanstd[k8], marker='o',
                 linestyle="none", label=f'{ptbins[k8]}'+r'$<p_t<$'+f'{ptbins[k8+1]}')
    
        ### Calculate resonable limits excluding the few points with insane errors
        yerr_norm = np.concatenate([np.sqrt(meanvar[[k2, k4, k6, k8]]) ])
        y_norm = np.concatenate([mean_p[[k2, k4, k6, k8]]])
        norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
        ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]) ,np.max((yerr_norm+y_norm)[norm_pos]))
        ax.set_xlabel(r'$\eta$');
        ax.set_ylabel(r'median response');
        ax.legend()
        if test_run:
            plt.savefig('test/fig/corr_vs_eta'+samp+tag_full+'_test.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
            plt.savefig('test/fig/corr_vs_eta'+samp+tag_full+'_test.png', dpi=plt.rcParamsDefault['figure.dpi']);
        else:
            plt.savefig('fig/corr_vs_eta'+samp+tag_full+'.pdf');
            plt.savefig('fig/corr_vs_eta'+samp+tag_full+'.png');
    
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
    
    medians = []
    medianstds = []
    
    combine_antiflavour = False
    subsamples = ['', '_b', '_c', '_ud', '_s', '_g', '_bbar', '_cbar', '_udbar', '_sbar']
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
            medians.append(median[0][0])
            medianstds.append(medianstd[0][0])
            for data, name in zip([mean, meanvar, median, medianstd],["Mean", "MeanVar", "Median", "MedianStd"]):
                save_data(data, name, samp)
                
        meanstd = np.sqrt(meanvar)
                
        if fine_etabins or one_bin:
            plot_corrections_eta(median, samp, medianstd)
        else:
            plot_corrections(mean, samp, meanstd)
    
    
    
    print('-----'*10)
    print("All done. Congrats!")
    
if __name__ == "__main__":
    main()