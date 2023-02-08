### CoffeaJERC-Andris.py
### File automatically converted using ConvertJupyterToPy.ipynb from CoffeaJERC-Andris.ipynb
### No comments or formatting is preserved by the transfer
def main():
    
    from notebook.services.config import ConfigManager
    c = ConfigManager()
    c.update('notebook', {"CodeCell": {"cm_config": {"autoCloseBrackets": False}}})
    
    import bokeh
    import time
    import copy
    import scipy.stats as ss
    from scipy.optimize import curve_fit
    from coffea import processor, nanoevents, util
    from coffea.nanoevents.methods import candidate
    from coffea.nanoevents import NanoAODSchema, BaseSchema
    
    import awkward as ak
    import numpy as np
    import glob as glob
    import itertools
    import pandas as pd
    from numpy.random import RandomState
    import importlib
    
    from dask.distributed import Client
    import inspect
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import hist
    
    from pltStyle import pltStyle
    import os
    
    UsingDaskExecutor = True
    CERNCondorCluster = False
    CoffeaCasaEnv     = False
    load_preexisting  = False    ### True if don't repeat the processing of files and use preexisting JER from output
    test_run          = True   ### True if run only on one file
    load_fit_res      = False   ### True if don't repeat the response fits
    
    fine_etabins      = False   ### Don't merge eta bins together when fitting responses. Preprocessing always done in many bins
    one_bin           = False   ### Unite all eta and pt bins in one
    
    Nfiles = 100                 ### -1 for all files
    
    tag = '_L5'                 ### L5 or L23, but L23 not supported since ages
    
    add_tag = '_LHEflav1_TTBAR-JME'
    
    xrootdstr = 'root://xrootd-cms.infn.it/'
    
    if CoffeaCasaEnv:
        xrootdstr = 'root://xcache/'
        
    prixydir = '/afs/cern.ch/user/a/anpotreb/k5-ca-proxy.pem'
        
    dataset = 'fileNames/fileNames_TTToSemi20UL18_JMENano.txt'
    
    with open(dataset) as f:
        rootfiles = f.read().split()
    
    Processor = importlib.import_module('CoffeaJERCProcessor'+tag).Processor
    
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
        
    if not os.path.exists("fig"):
        os.mkdir("fig/")
        os.mkdir("fig/responses/")
        
    if test_run and not os.path.exists("test"):
        os.mkdir("test/")
        os.mkdir("test/out_txt")
        os.mkdir("test/fig")
        
    if test_run:
        Nfiles = 1
        
    fileslist = [xrootdstr + file for file in rootfiles]
    fileslist = fileslist[:Nfiles] # if add_tag=='QCD' else fileslist # :20 to skim the events
        
    print(f'Runing on dataset {dataset}\n Number of files: {Nfiles}\n Job with the full tag {tag_full}\n Outname = {outname}')
    
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
    
    combineTTbar = False
    
    file_tags = ['Semi', 'Dilep', 'Had']
    
    filesets = {}
    if combineTTbar:
        for ftag in file_tags:
            data_name = f'fileNames/fileNames_TTTo{ftag}20UL18_JMENano.txt'
            with open(data_name) as f:
                rootfiles = f.read().split()
            fileslist = [xrootdstr + file for file in rootfiles]
            fileslist = fileslist[:Nfiles]
            xsec = find_xsec(data_name)
            filesets[ftag] = {"files": fileslist, "metadata": {"xsec": xsec}}
    else:
        filesets = {'QCD': {"files": fileslist, "metadata": {"xsec": xsec}}}
    
    import os
    
    os.environ['X509_USER_PROXY'] = '/afs/cern.ch/user/a/anpotreb/k5-ca-proxy.pem'
    if os.path.isfile(os.environ['X509_USER_PROXY']):
        print("Found proxy at {}".format(os.environ['X509_USER_PROXY']))
    else:
        print("os.environ['X509_USER_PROXY'] ",os.environ['X509_USER_PROXY'])
    os.environ['X509_CERT_DIR'] = '/cvmfs/cms.cern.ch/grid/etc/grid-security/certificates'
    os.environ['X509_VOMS_DIR'] = '/cvmfs/cms.cern.ch/grid/etc/grid-security/vomsdir'
    os.environ['X509_USER_CERT'] = '/afs/cern.ch/user/a/anpotreb/k5-ca-proxy.pem'
    
    import uproot
    
    ff = uproot.open(fileslist[0])
    ff.keys()
    ff.close()
    
    if(UsingDaskExecutor and CoffeaCasaEnv):
        client = Client("tls://ac-2emalik-2ewilliams-40cern-2ech.dask.coffea.casa:8786")
        client.upload_file('CoffeaJERCProcessor.py')
    
    env_extra = [
                'export XRD_RUNFORKHANDLER=1',
                f'export X509_USER_PROXY=/afs/cern.ch/user/a/anpotreb/k5-ca-proxy.pem',
                f'export X509_CERT_DIR={os.environ["X509_CERT_DIR"]}',
            ]
    
    if(UsingDaskExecutor and not CoffeaCasaEnv):
        from dask.distributed import Client 
        if not CERNCondorCluster:
            client = Client()
            client.get_versions(check=True)
    
        else:
            from dask_lxplus import CernCluster
            import socket
    
            cluster = CernCluster(
                env_extra=env_extra,
                cores = 4,
                memory = '4000MB',
                disk = '2000MB',
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
                    'MY.JobFlavour': '"longlunch"',
                    'transfer_input_files': '/afs/cern.ch/user/a/anpotreb/top/JERC/JMECoffea/count_2d.py',
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
    
    if not load_preexisting:
        if not UsingDaskExecutor:
            chosen_exec = 'futures'
            output = processor.run_uproot_job(filesets,
                                              treename='Events',
                                              processor_instance=Processor(),
                                              executor=processor.iterative_executor,
        #                                        executor=processor.futures_executor,
                                              executor_args={
                                                  'skipbadfiles':True,
                                                  'schema': NanoAODSchema, #BaseSchema
                                                  'workers': 2},
                                              chunksize=Chunk[0])#, maxchunks=Chunk[1])
        else:
            chosen_exec = 'dask'
            output = processor.run_uproot_job(filesets,
                                              treename='Events',
                                              processor_instance=Processor(),
                                              executor=processor.dask_executor,
                                              executor_args={
                                                  'client': client,
                                                  'skipbadfiles':True,
                                                  'schema': NanoAODSchema, #BaseSchema
                                                  'xrootdtimeout': 60,
                                                  'retries': 2,
                                              },
                                              chunksize=Chunk[0])#, maxchunks=Chunk[1])
    
        elapsed = time.time() - tstart
        print("Processor finished. Time elapsed: ", elapsed)
        print("Saving the output histograms under: ", outname)
        util.save(output, outname)
    else:
        output = util.load(outname)
        print("Loaded histograms from: ", outname)
    
    if UsingDaskExecutor:
        client.close()
        time.sleep(5)
        if CERNCondorCluster or CoffeaCasaEnv:
            cluster.close()
    
    if UsingDaskExecutor:
        client.close()
        time.sleep(5)
        if CERNCondorCluster or CoffeaCasaEnv:
            cluster.close()
    
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
    if fine_etabins==True:
        ptbins = np.array([15, 40, 150, 400, 4000, 10000])
        ptbins_c = (ptbins[:-1]+ptbins[1:])/2
        etabins = output['ptresponse'].axes["jeteta"].edges #output['ptresponse'].axis('jeteta').edges()
    elif one_bin==True:
        ptbins = np.array([15, 10000])
        ptbins_c = (ptbins[:-1]+ptbins[1:])/2
        etabins = np.array([-5, -3, -2.5, -1.3, 0, 1.3, 2.5, 3, 5])
        etabins = np.array([etabins[0], 0, etabins[-1]])
    else:
        ptbins = output['ptresponse'].axes["pt_gen"].edges 
        ptbins_c = output['ptresponse'].axes['pt_gen'].centers
        etabins = np.array([-5, -3, -2.5, -1.3, 0, 1.3, 2.5, 3, 5])
    
        
    jetpt_length = len(ptbins)-1
    jeteta_length = (len(etabins)-1)//2
    
    etabins_mod = etabins[(len(etabins)-1)//2:]
    etabins_c = (etabins_mod[:-1]+etabins_mod[1:])/2 #output['ptresponse'].axis('jeteta').centers()
    
    ptresp_edd = output['ptresponse'].axes['ptresponse'].edges
    plot_pt_edges = ptresp_edd[0:np.nonzero(ptresp_edd>=2.0)[0][0]]
    hist_pt_edges = plot_pt_edges[1:-1]   #for plotting. To exclude overflow from the plot
    
    from pltStyle import pltStyle
    pltStyle(style='paper')
    plt.rcParams['figure.subplot.left'] = 0.162
    plt.rcParams['figure.dpi'] = 150
    
    if combineTTbar:
        ids = output[list(output.keys())[0]].axis('dataset').identifiers()
        names = [idii.name for idii in ids  ]
    
        output_comb = {}
    
        N = {}
        for s in names:
            N[s] = output['cutflow'][s+': all events']
        N_av = sum(N.values())/3
    
        for key in output.keys():
            if key!='cutflow':
                hist_comb = output[key].integrate('dataset', ids[0])
                hist_comb.scale(find_xsec(names[0])*N[names[0]]/N_av)
                for ii in range(1,len(ids)-1):
                    hist2 = output[key].integrate('dataset', ids[ii])
                    hist2.scale(find_xsec(ids[ii].name)*N[names[ii]]/N_av)
                    hist_comb = hist_comb+hist2
                output_comb[key] = hist_comb
            else:
                cut_keys = list(output[key].keys())
                len_new_keys = len(cut_keys)//3
                output_comb["cutflow"] = {}
                for cut in range(len_new_keys):
                    output_comb["cutflow"]["Inclusive"+cut_keys[cut][4:]] = (output[key][cut_keys[cut]]*find_xsec(names[0])*N[names[0]]/N_av +
                                                                   output[key][cut_keys[cut+len_new_keys]]*find_xsec(names[1])*N[names[1]]/N_av +
                                                                   output[key][cut_keys[cut+2*len_new_keys]]*find_xsec(names[2])*N[names[2]]/N_av 
                                                                  )
                    
        output = output_comb
        tag_full = tag + '_LHEflav1_TTBAR-Inclusive-JME'
    
        
        
    
            
    
    def get_median(yvals, bin_edges):
        yvals_cumsum = np.cumsum(yvals)
        N = np.sum(yvals)
        med_bin = np.nonzero(yvals_cumsum>N/2)[0][0] if N>200 else 0
        
        median = bin_edges[med_bin] + (N/2 - yvals_cumsum[med_bin-1])/yvals[med_bin]*(bin_edges[med_bin+1]
                                                                                     - bin_edges[med_bin])
        return median
    
    def fit_response(xvals, yvals, N):
        if_failed = False
        
        nonzero_bins = np.sum(yvals>0)
        if nonzero_bins<2 or N<50:
            p2=[0,0,0]
            chi2 = np.nan
            cov = np.array([[np.nan]*3]*3)
            Ndof = 0
        #                 print("Too little data points, skipping p = ", p2)
        else:
            try:
                p, cov = curve_fit(gauss, xvals, yvals, p0=[10,1,1])
                     ######## Second Gaussian ########
                xfit_l = np.where(xvals>=p[1]-np.abs(p[2])*1.5)[0][0]
                xfit_hs = np.where(xvals>=p[1]+np.abs(p[2])*1.5)[0]
                xfit_h = xfit_hs[0] if len(xfit_hs)>0 else len(xvals)
        #                     print("xfit_l = ", xfit_l, ", xfit_h = ", xfit_h)
    
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
        #                     if ptBin.lo>290:
        #                         print("xfit_l = ", xfit_l, ", h = ", xfit_h)
        #                         print("yvals = ", yvals)
        #                         print("yvals2 = ", yvals2)
                p2, cov = curve_fit(gauss, xvals2, yvals2, p0=p)
                             ######## End second Gaussian ########
    
                ygaus = gauss(xvals, *p2)
                chi2 = sum((yvals-ygaus)**2/(yvals+1E-9))
                Ndof = len(xvals2)-3
        #                     if chi2<50000:
        #                         pass
        #                         print("Fit converged, p = ", p2, ", chi2 = ", chi2 )
        #                     else:
        #                         print("Fit failed because of high chi2, p = ", p2, ", chi2 = ", chi2 )
            except(RuntimeError):   #When fit failed
                p2=[0,0,0]
        #                     print("Fit failed because of non-convergance, p = ", p2)
                chi2 = np.nan
                cov = np.array([[np.nan]*3]*3)
                Ndof = 0
                if_failed = True
                
        return [p2, cov, chi2, Ndof, if_failed]
    
    def plot_response_dist(histo, xvals, p2, cov, chi2, Ndof, median, medianstd, N, figName ):
        width_ik = np.abs(p2[2])
        f_xvals = np.linspace(0,max(xvals),5001)
        fgaus2 = gauss(f_xvals, *p2)
        edd = histo.axes['ptresponse'].edges
        histo = histo[1:len(edd)-2] 
        #remove the underflow, overflow. Not sure if in hist.hist it is stored in the last and first bin like in coffea.hist
        
        fig, ax2 = plt.subplots();
        histo.plot1d(ax=ax2, overlay='dataset', histtype='fill', alpha=0.6)
        # ax2.plot(f_xvals, fgaus, label='Gaus',linewidth=1.8)
        ax2.plot(f_xvals, fgaus2, label='Gaus',linewidth=1.8)
        ax2.set_xlabel("Response ($p_{T,reco}/p_{T,ptcl}$)")
        ax2.set_xlim(plot_pt_edges[[0,-1]])
        h = ax2.get_ylim()[1]/1.05
        plt.text(0.03,0.95*h,r'Mean {0:0.3f}$\pm${1:0.3f}'.format(p2[1], np.sqrt(cov[1,1])))
        plt.text(0.03,0.88*h,r'Width {0:0.3f}$\pm${1:0.3f}'.format(width_ik, np.sqrt(cov[2,2])))
        plt.text(0.03,0.81*h,r'Median {0:0.3f}$\pm${1:0.3f}'.format(median, medianstd))
        plt.text(0.03,0.73*h,r'$\chi^2/ndof$ {0:0.2g}/{1:0.0f}'.format(chi2, Ndof))
        plt.text(0.03,0.66*h,r'N data = {0:0.3g}'.format(N))
        ax2.legend();
    
        
        plt.savefig(figName+'.png', dpi=plt.rcParamsDefault['figure.dpi']);
        plt.savefig(figName+'.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
        plt.close();   
    
    import warnings
    barable_samples = ['b', 'c', 's', 'u', 'd']
    
    def fit_responses(output, samp='_b'):
        warnings.filterwarnings('ignore')
        saveplots = False
        if test_run or fine_etabins:
            saveplots = False
        saveplots = True
    
        
        response_hist_all_samp = output['ptresponse'][{"pt_reco":sum}]
        recopt_hist_all_samp = output['ptresponse'][{"ptresponse":sum}]
        
        if combine_antiflavour and (samp in barable_samples):
            response_hist = response_hist_all_samp[{"jet_flav":samp}] + response_hist_all_samp[{"jet_flav":samp+'bar'}]
            recopt_hist = recopt_hist_all_samp[{"jet_flav":samp}] + recopt_hist_all_samp[{"jet_flav":samp+'bar'}]
        else:
            response_hist = response_hist_all_samp[{"jet_flav":samp}]
            recopt_hist = recopt_hist_all_samp[{"jet_flav":samp}]
    
        results = {}
        results["Mean"] = np.zeros((jetpt_length, jeteta_length))
        results["Median"] = np.zeros((jetpt_length, jeteta_length))
        results["MedianStd"] = np.zeros((jetpt_length, jeteta_length))
        results["MeanVar"] = np.zeros((jetpt_length, jeteta_length))
        results["MeanRecoPt"] = np.zeros((jetpt_length, jeteta_length))
    
        N_converge = 0
        N_not_converge = 0
    
        FitFigDir1 = 'fig/responses/responses'+tag_full
        if saveplots and not os.path.exists(FitFigDir1):
            os.mkdir(FitFigDir1)
        
        FitFigDir = FitFigDir1+'/response_pt_eta'+samp+tag_full
        if saveplots and not os.path.exists(FitFigDir):
            os.mkdir(FitFigDir)
    
        xvals = response_hist.axes['ptresponse'].centers[1:] #[1:] to exclude the second peak for low pt
        response_edges = response_hist.axes['ptresponse'].edges[1:]
    
        for i in range(jetpt_length):
            pt_lo = ptbins[i]
            pt_hi = ptbins[i+1]
        #         print('-'*25)
    
            if not np.isinf(pt_hi):
                pt_string = '_pT'+str(int(pt_lo))+'to'+str(int(pt_hi))
            else:
                pt_string = '_pT'+str(pt_lo) + 'to' + str(pt_hi)
                pt_string = pt_string.replace('.0','').replace('-infto','0to')
    
            for k in range(jeteta_length):
                etaPl_lo = etabins[k+jeteta_length]
                etaPl_hi = etabins[k+1+jeteta_length]
                etaMi_lo = etabins[jeteta_length-k-1]
                etaMi_hi = etabins[jeteta_length-k]
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
    
                yvals = histo.values()[1:]     #[1:] to exclude the second peak for low pt
                N = np.sum(yvals)
    
                ####################### Calculate median and rms ############################
                yvals_cumsum = np.cumsum(yvals)
                med_bin = np.nonzero(yvals_cumsum>N/2)[0][0] if N>200 else 0
                median = response_edges[med_bin] + (N/2 - yvals_cumsum[med_bin-1])/yvals[med_bin]*(response_edges[med_bin+1]
                                                                                          - response_edges[med_bin])
    
                hist_mean = np.sum(xvals*yvals)/sum(yvals) 
                hist_rms = np.sqrt(np.sum(yvals*((hist_mean-xvals)**2))/sum(yvals))
                medianstd = 1.253 * hist_rms/np.sqrt(N)
                
                ##################### Mean of the pt_reco  ######################
                ### (this is the binned mean, not the real data mean)  
                y = histopt.counts()
                x = histopt.axes['pt_reco'].centers
                mean_reco_pt = np.sum(x*y)/sum(y)
    
                ####################### Fitting ############################
                p2, cov, chi2, Ndof, if_failed = fit_response(xvals, yvals, N)
                if if_failed:
                    N_not_converge += 1
                else:
                    N_converge += 1
                
                results["Mean"][i,k] = p2[1]
                results["MeanVar"][i,k] = cov[1,1]
                results["Median"][i,k] = median
                results["MedianStd"][i,k] = medianstd
                results["MeanRecoPt"][i,k] = mean_reco_pt
        #             chi2s[i,k] = chi2
    
        ####################### Plotting ############################
                if  saveplots:
                    figName = FitFigDir+'/ptResponse'+pt_string+eta_string
                    plot_response_dist(histo, xvals, p2, cov, chi2, Ndof, median, medianstd, N, figName)              
    
        print("N converge = ", N_converge, "N_not_converge = ", N_not_converge );
        warnings.filterwarnings('default')
        
        return results  
    
    def plot_corrections(mean, meanstd, samp):
        ### To ignore the points with 0 on y axis when setting the y axis limits
        mean_p = mean.copy()
        mean_p[mean_p==0] = np.nan
    
        fig, ax = plt.subplots()
        start = np.searchsorted(ptbins_c, 20, side='left') #np.where(ptbins<=20)[0][-1]
        
        k2 = np.where(etabins_mod<=0)[0][-1]
        k4 = np.where(etabins_mod<=1.3)[0][-1]
        k6 = np.where(etabins_mod<=2.5)[0][-1]
        k8 = np.where(etabins_mod<=3.0)[0][-1]
        lastbin = np.where(~ np.isnan(mean_p[:, k2]*mean_p[:, k4]*mean_p[:, k6]*mean_p[:, k8]))[0][-1]
        
        ptbins_plot = ptbins_c[start:lastbin]
        meanstd = meanstd[start:lastbin,:]
        
        mean_p = mean_p[start:lastbin]
        
        plt.errorbar(ptbins_plot, mean_p[:,k2], yerr=meanstd[:,k2], marker='o',
                     linestyle="none", label=f'{etabins_mod[k2]}'+r'$<\eta<$'+f'{etabins_mod[k2+1]}')
        plt.errorbar(ptbins_plot, mean_p[:,k4], yerr=meanstd[:,k4], marker='o',
                 linestyle="none", label=f'{etabins_mod[k4]}'+r'$<\eta<$'+f'{etabins_mod[k4+1]}')
        plt.errorbar(ptbins_plot, mean_p[:,k6], yerr=meanstd[:,k6], marker='o',
                 linestyle="none", label=f'{etabins_mod[k6]}'+r'$<\eta<$'+f'{etabins_mod[k6+1]}')
        plt.errorbar(ptbins_plot, mean_p[:,k8], yerr=meanstd[:,k8], marker='o',
                 linestyle="none", label=f'{etabins_mod[k8]}'+r'$<\eta<$'+f'{etabins_mod[k8+1]}')
    
        ### Calculate resonable limits excluding the few points with insane errors
        yerr_norm = np.concatenate([meanstd[:,[k2, k4, k6, k8]] ])
        y_norm = np.concatenate([mean_p[:,[k2, k4, k6, k8]]])
        norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
        
        left_lim = np.min((y_norm-yerr_norm)[norm_pos])
        right_lim = np.max((yerr_norm+y_norm)[norm_pos])
        lim_pad = (right_lim - left_lim)/20
        ax.set_ylim(left_lim-lim_pad, right_lim+lim_pad)
        
        ax.set_xscale('log')
        
        
        
        good_xlims = ax.get_xlim()
        ax.set_xticks([20, 50, 100, 500, 1000, 5000])
        ax.set_xlim(good_xlims)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel(r'$p_T$ (GeV)');
        ax.set_ylabel(r'median response');
        ax.legend()
        if test_run:
            plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
            plt.savefig('test/fig/corr_vs_pt'+samp+tag_full+'_test.png', dpi=plt.rcParamsDefault['figure.dpi']);
        else:
            plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.pdf');
            plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.png');
    
        plt.show();
        
    
    def plot_corrections_eta(mean, meanstd, samp):
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
    
    combine_antiflavour = True
    # subsamples = ['', '_b', '_c', '_u', '_d', '_s', '_g', '_bbar', '_cbar', '_ubar', '_dbar','_sbar']
    subsamples = ['b', 'c', 'u', 'd', 's', 'g', 'bbar', 'cbar', 'ubar', 'dbar','sbar']
    subsamples = ['b']
    # subsamples = ['_b', '_c', '_ud', '_s', '_g']
    # subsamples = ['', '_ud', '_g'] # , '_b']
    for samp in subsamples:
        print('-'*25)
        print('-'*25)
        print('Fitting subsample: ', samp)
        if load_fit_res:
            samp = "_"+samp
            result = {}
            keys = ["Median", "MedianVar", "Median", "MedianStd", "MeanRecoPt"] 
            for key in keys:
                result[key] = read_data(key, samp)

        else:
            result = fit_responses(output, samp)
            samp = "_"+samp
            medians.append(result["Median"][0][0])
            medianstds.append(result["MedianStd"][0][0])
            for key in result:
                save_data(result[key], key, samp)
                pass

        median = result["Median"]
        medianStd = result["MedianStd"]

        meanstd = np.sqrt(result["MeanVar"])

        if fine_etabins or one_bin:
            plot_corrections_eta(result["Median"], result["MedianStd"], samp)
        else:
            plot_corrections(result["Median"], result["MedianStd"], samp)
    
    print('-----'*10)
    print("All done. Congrats!")
    
if __name__ == "__main__":
    main()