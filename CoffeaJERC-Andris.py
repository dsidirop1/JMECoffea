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
    
    from pltStyle import pltStyle
    import os
    
    from CoffeaJERCProcessor_L5 import Processor
    
    UsingDaskExecutor = True
    CoffeaCasaEnv     = False
    load_preexisting  = True    ### True if don't repeat the processing of files and use preexisting JER from output
    test_run          = False     ### True if run only on one file
    
    tag = '_L5'
    
    exec('from CoffeaJERCProcessor'+tag+' import Processor') 
    
    add_tag = '_QCD' # '_QCD' # '_testing_19UL18' # ''
    tag_full = tag+add_tag
    
    outname = 'out/CoffeaJERCOutputs'+tag_full+'.coffea'
    outname = outname+'_test' if test_run else outname
    
    xrootdstr = 'root://cmsxrootd.fnal.gov/'
    
    if CoffeaCasaEnv:
        xrootdstr = 'root://xcache/'
        
    dataset = 'fileNames_QCD20UL18.txt' # 'dataset.txt'  #
    
    rootfiles = open(dataset).read().split()
    
    fileslist = [xrootdstr + file for file in rootfiles]
    
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
    
    print("Output:")
    print(output)
    
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
    f_xvals = np.linspace(0,5,5001)
    ptbins = output['ptresponse'].axis('pt').edges()
    etabins = output['ptresponse'].axis('jeteta').edges()
    jetpt_length = len(ptbins)-1
    jeteta_length = (len(etabins)-1)//2
    
    etabins_mod = etabins[(len(etabins)-1)//2:]
    
    from pltStyle import pltStyle
    pltStyle(style='paper')
    plt.rcParams['figure.subplot.left'] = 0.160
    
    import warnings
    
    def fit_responses(output, samp='_b'):
        warnings.filterwarnings('ignore')
    
        mean = np.zeros((jetpt_length, jeteta_length))
        median = np.zeros((jetpt_length, jeteta_length))
        width = np.zeros((jetpt_length, jeteta_length))
        chi2s = np.zeros((jetpt_length, jeteta_length))
        meanvar = np.zeros((jetpt_length, jeteta_length))
        
        N_converge = 0
        N_not_converge = 0
    
        FitFigDir = 'fig/response_pt_eta'+samp+tag_full
        if not os.path.exists(FitFigDir):
            os.mkdir(FitFigDir)
        plot_pt_edges = output['ptresponse'+samp].axis('ptresponse').edges()[0:42] ##Put plotting limits to the histogram
        hist_pt_edges = plot_pt_edges[1:-1]   #for plotting. To explude overflow from the plot
        plot_response_axis = hist.Bin("jeteta", r"Jet $\eta$", hist_pt_edges)
    
        xvals = output['ptresponse'+samp].axis('ptresponse').centers()[1:] #[1:] to exclude the second peak for low pt
        f_xvals = np.linspace(0,5,5001)
    
        for i in range(len(ptbins)-1):
            ptBin = hist.Interval(ptbins[i], ptbins[i+1])
            print('-'*25)
            print('pt bin '+str(ptBin))
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
                histvals = np.repeat(histo.axis('ptresponse').centers(), np.array(histo.values()[('QCD',)],dtype='int'))
                yvals = histo.values()[('QCD',)][1:]  #[1:] to exclude the second peak for low pt
    
                try:
                    p, arr = curve_fit(gauss, xvals, yvals, p0=[10,1,1])
                    N_converge += 1
    
                                ### Second Gaussian
                    xfit_l = np.where(xvals>=p[1]-np.abs(p[2])*1.5)[0][0]
                    xfit_h = np.where(xvals>=p[1]+np.abs(p[2])*1.5)[0][0]
                    if len(range(xfit_l,xfit_h))<4: #if there are only 3pnts, the uncertainty is infty
                        xfit_l = xfit_l-1
                        xfit_h = xfit_h+1
                        if len(range(xfit_l,xfit_h))<4:
                            xfit_l = xfit_l-1
                            xfit_h = xfit_h+1
                    xvals2 = xvals[xfit_l: xfit_h]
                    yvals2 = yvals[xfit_l: xfit_h]
                    p2, arr = curve_fit(gauss, xvals2, yvals2, p0=p)
    
                    ygaus = gauss(xvals, *p2)
                    chi2 = sum((yvals-ygaus)**2/(yvals+1E-9))
                    Ndof = len(xvals2)-3
                    if chi2<50000:
                        pass
                        print("Fit converged, p = ", p2, ", chi2 = ", chi2 )
                    else:
                        print("Fit failed because of high chi2, p = ", p2, ", chi2 = ", chi2 )
                except:
        #             print("Fit failed because of non-convergance, p = ", p)
                    N_not_converge += 1
                    continue
    
                fgaus2 = gauss(f_xvals, *p2)
    
                median[i,k] = np.median(histvals)
                mean[i,k] = p2[1] # - (scal-1)
                width[i,k] = p2[2]
                chi2s[i,k] = chi2
                meanvar[i,k] = arr[1,1]
    
       ####################### Plotting ############################
                if not test_run:
                    N = histo.integrate('ptresponse').values()[('QCD',)]
                    histo = histo.rebin('ptresponse', plot_response_axis)
    
#                     h = np.max(yvals[:-1]);
#                     h = h if h!=0 else 0.05
                    fig, ax2 = plt.subplots();
                    hist.plot1d(histo, ax=ax2, overlay='dataset', overflow='all',
                                fill_opts={'alpha': .5, 'edgecolor': (0,0,0,0.3), 'linewidth': 1.4})
                    # ax2.plot(f_xvals, fgaus, label='Gaus',linewidth=1.8)
                    ax2.plot(f_xvals, fgaus2, label='Gaus',linewidth=1.8)
                    ax2.set_xlabel("Response ($E_{RECO}/E_{GEN}$)")
                    ax2.set_xlim(plot_pt_edges[[0,-1]])
                    h = ax2.get_ylim()[1]/1.05
                    plt.text(0.03,0.95*h,'Mean {0:0.3f}$\pm${1:0.3f}'.format(p2[1],np.sqrt(arr[1,1])))
                    plt.text(0.03,0.88*h,'Width {0:0.3f}$\pm${1:0.3f}'.format(p2[2],np.sqrt(arr[2,2])))
                    plt.text(0.03,0.81*h,'Median {0:0.3f}'.format(np.median(histvals)))
                    plt.text(0.03,0.73*h,'$\chi^2/ndof$ {0:0.2g}/{1:0.0f}'.format(chi2, Ndof))
                    plt.text(0.03,0.66*h,'N data = {0:0.3g}'.format(N))
                    ax2.legend();
    
                    plt.savefig(FitFigDir+'/ptResponse'+pt_string+eta_string+'.png');
                    plt.savefig(FitFigDir+'/ptResponse'+pt_string+eta_string+'.pdf');
                    plt.close();                
    
        warnings.filterwarnings('default')
        print("N converge = ", N_converge, "N_not_converge = ", N_not_converge );
        warnings.filterwarnings('default')
        
        return [mean, width, median, chi2s, meanvar]
        
    
    etabins_mod
    
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
    
        
        # ax.plot(ptbins[start:],mean_p[start:,k0], 'o', label=f'${etabins[k0]}<\eta<{etabins[k0+1]}$')
        plt.errorbar(ptbins[start:-1],mean_p[start:,k2], yerr=np.sqrt(meanvar[start:,k2]), marker='o',
                     linestyle="none", label=f'${etabins_mod[k2]}<\eta<{etabins_mod[k2+1]}$')
        plt.errorbar(ptbins[start:-1],mean_p[start:,k4], yerr=np.sqrt(meanvar[start:,k4]), marker='o',
                 linestyle="none", label=f'${etabins_mod[k4]}<\eta<{etabins_mod[k4+1]}$')
        plt.errorbar(ptbins[start:-1],mean_p[start:,k6], yerr=np.sqrt(meanvar[start:,k6]), marker='o',
                 linestyle="none", label=f'${etabins_mod[k6]}<\eta<{etabins_mod[k6+1]}$')
        plt.errorbar(ptbins[start:-1],mean_p[start:,k8], yerr=np.sqrt(meanvar[start:,k8]), marker='o',
                 linestyle="none", label=f'${etabins_mod[k8]}<\eta<{etabins_mod[k8+1]}$')
        # ax.plot(ptbins[start:],mean_p[start:,k3], 'o', label=f'${etabins[k3]}<\eta<{etabins[k3+1]}$')
    
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
            plt.savefig('fig//corr_vs_pt'+samp+tag_full+'_test.pdf');
            plt.savefig('fig/corr_vs_pt'+samp+tag_full+'_test.png');
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
            df.to_csv('out_txt/EtaBinsvsPtBins'+name+tag+'_test.csv')
    
    def read_data(name, samp):
        if not test_run:
            df_csv = pd.read_csv('out_txt/EtaBinsvsPtBins'+name+samp+tag_full+'.csv').set_index('etaBins')
        else:
            df_csv = pd.read_csv('out_txt/EtaBinsvsPtBins'+name+tag+'_test.csv').set_index('etaBins')
        
        data = df_csv.to_numpy().transpose()
        return data
    
    load_fit_res = False
    subsamples = ['', '_b', '_c', '_l', '_g']
#     subsamples = ['']
    for samp in subsamples:
        print('-'*25)
        print('-'*25)
        print('Fitting subsample: ', samp)
        if load_fit_res:
            mean = read_data("Mean", samp)
            meanvar = read_data("MeanVar", samp)
        else:
            mean, width, median, chi2s, meanvar = fit_responses(output, samp)
            for data, name in zip([mean, width, median, meanvar],["Mean", "Width", "Median", "MeanVar"]):
                save_data(data, name, samp)
                
        plot_corrections(mean, samp, meanvar)
    
    
if __name__ == "__main__":
    main()