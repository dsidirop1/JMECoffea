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
    load_preexisting  = False    ### True if don't repeat the processing of files and use preexisting JER from output
    test_run          = True     ### True if run only on one file
    
    tag = '_L5'
    
    exec('from CoffeaJERCProcessor'+tag+' import Processor') 
    
    add_tag = '' # '_testing_19UL18' # ''
    tag_full = tag+add_tag
    
    outname = 'out/CoffeaJERCOutputs'+tag_full+'.coffea'
    outname = outname+'_test' if test_run else outname
    
    xrootdstr = 'root://cmsxrootd.fnal.gov/'
    
    if CoffeaCasaEnv:
        xrootdstr = 'root://xcache/'
        
    dataset = 'dataset.txt'
    
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
        
    print("Output:")
    print(output)
    elapsed = time.time() - tstart
    
    output
    
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
    from pltStyle import pltStyle
    pltStyle(style='paper')
    plt.rcParams['figure.subplot.left'] = 0.160
    
    output
    
    import warnings
    
    f_xvals = np.linspace(0,5,5001)
    
    def fit_responses(output, samp='_b'):
        warnings.filterwarnings('ignore')
        ptbins = output['ptresponse'].axis('pt').edges()
        etabins = output['ptresponse'].axis('jeteta').edges()[:-1]
        jetpt_length = len(ptbins)
        jeteta_length = len(etabins)
    
        mean = np.zeros((jetpt_length, jeteta_length))
        median = np.zeros((jetpt_length, jeteta_length))
        width = np.zeros((jetpt_length, jeteta_length))
        
        
        N_converge = 0
        N_not_converge = 0
        chi2s = np.zeros((jetpt_length, jeteta_length))
    
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
    
            for k in range(len(etabins)-1):
    
                etaBin = hist.Interval(etabins[k], etabins[k+1])
                print("etaBin = ", etaBin ,", ptBin = ", ptBin )
                eta_string = '_eta'+str(etaBin.lo)+'to'+str(etaBin.hi)
                eta_string = eta_string.replace('.','')
                
                # The name integrate is a bit misleasding in this line. Is there another way to "slice" a histogram? //Andris
                histo = output['ptresponse'+samp].integrate('jeteta', etaBin).integrate('pt', ptBin) 
                histvals = np.repeat(histo.axis('ptresponse').centers(), np.array(histo.values()[('QCD',)],dtype='int'))
                yvals = histo.values()[('QCD',)][1:]  #[1:] to exclude the second peak for low pt
    
                try:
                    p, arr = curve_fit(gauss, xvals, yvals, p0=[10,1,1])
                    N_converge += 1
    
                                ### Second Gaussian
                    xfit_l = np.where(xvals>=p[1]-np.abs(p[2])*1.5)[0][0]
                    xfit_h = np.where(xvals>=p[1]+np.abs(p[2])*1.5)[0][0]
                    xvals2 = xvals[xfit_l: xfit_h]
                    yvals2 = yvals[xfit_l: xfit_h]
                    p2, arr = curve_fit(gauss, xvals2, yvals2, p0=p)
    
                    ygaus = gauss(xvals, *p2)
                    chi2 = sum((yvals-ygaus)**2/(yvals+1E-9))
                    Ndof = len(xvals)-3
                    if chi2<50000:
                        pass
                        print("Fit converged, p = ", p2, ", chi2 = ", chi2 )
                    else:
                        print("Fit failed because of high chi2, p = ", p2, ", chi2 = ", chi2 )
                except:
        #             print("Fit failed because of non-convergance, p = ", p)
                    N_not_converge += 1
                    continue
    
                fgaus = gauss(f_xvals, *p)
                fgaus2 = gauss(f_xvals, *p2)
    
                median[i,k] = np.median(histvals)
                mean[i,k] = p[1] # - (scal-1)
                width[i,k] = p[2]
                chi2s[i,k] = chi2
    
       ####################### Plotting ############################
                if not test_run:
                    N = histo.integrate('ptresponse').values()[('QCD',)]
                    histo = histo.rebin('ptresponse', plot_response_axis)
    
                    h = np.max(histo.values()[('QCD',)]);
                    h = h if h!=0 else 0.05
                    fig, ax2 = plt.subplots();
                    hist.plot1d(histo, ax=ax2, overlay='dataset', overflow='all',
                                fill_opts={'alpha': .5, 'edgecolor': (0,0,0,0.3), 'linewidth': 1.4})
                    ax2.plot(f_xvals, fgaus, label='Gaus',linewidth=1.8)
                    ax2.plot(f_xvals, fgaus2, label='Gaus, 2nd fit',linewidth=1.8)
                    ax2.set_xlabel("Response ($E_{RECO}/E_{GEN}$)")
                    ax2.set_xlim(plot_pt_edges[[0,-1]])
                    plt.text(1.4,0.75*h,'Mean {0:0.2f}'.format(p[1]))
                    plt.text(1.4,0.68*h,'Median {0:0.2f}'.format(np.median(histvals)))
                    plt.text(1.4,0.61*h,'Width {0:0.2f}'.format(p[2]))
                    plt.text(1.3,0.53*h,'$\chi^2/ndof$ {0:0.1f}/{0:0.1f}'.format(chi2, Ndof))
                    plt.text(1.3,0.46*h,'N data = {0:0.0f}'.format(N))
                    ax2.legend(ncol=2);
    
                    plt.savefig(FitFigDir+'/ptResponse'+pt_string+eta_string+'.png');
                    plt.savefig(FitFigDir+'/ptResponse'+pt_string+eta_string+'.pdf');
                    plt.close();                
    
        warnings.filterwarnings('default')
        print("N converge = ", N_converge, "N_not_converge = ", N_not_converge );
        warnings.filterwarnings('default')
        
    
        
    
    subsamples = ['', '_b']
    
    fit_responses(output, '')
    
    for sam in subsamples:
        print('-'*25)
        print('-'*25)
        print('Fitting subsample: ', sam)
        fit_responses(output, sam)
    
    samp = ''
    
    scalings = pd.read_csv('out_txt/EtaBinsvsPtBinsMean_L5_scale.csv').set_index('etaBins')
    
            
        
        
            
       
            
    
            
            
                
                
                
    
            
            
            
            
    
                
    
            
    
    mean_p = mean.copy()
    mean_p[mean_p==0] = np.nan
    
    fig, ax = plt.subplots()
    start = 17
    k0 = np.where(etabins<=-5)[0][-1]
    k4 = np.where(etabins<=-3)[0][-1]
    k1 = np.where(etabins<=-2.5)[0][-1]
    k5 = np.where(etabins<=-1.3)[0][-1]
    k2 = np.where(etabins<=0)[0][-1]
    k3 = np.where(etabins<=1.3)[0][-1]
    
    ax.plot(ptbins[start:],mean_p[start:,k4], 'o', label=f'${etabins[k4]}<\eta<{etabins[k4+1]}$')
    ax.plot(ptbins[start:],mean_p[start:,k1], 'o', label=f'${etabins[k1]}<\eta<{etabins[k1+1]}$')
    ax.plot(ptbins[start:],mean_p[start:,k5], 'o', label=f'${etabins[k5]}<\eta<{etabins[k5+1]}$')
    
    ax.set_xlabel(r'$p_T$ (GeV)');
    ax.set_ylabel(r'mean response');
    ax.set_xscale('log')
    ax.legend()
    if test_run:
        plt.savefig('fig/corr_vs_pt'+samp+tag_full+'_test.pdf');
    else:
        plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.pdf');
    
    plt.show();
    
    mean_p = mean.copy()
    mean_p[mean_p==0] = np.nan
    
    fig, ax = plt.subplots()
    start = 17
    k0 = np.where(etabins<=-5)[0][-1]
    k4 = np.where(etabins<=-3)[0][-1]
    k1 = np.where(etabins<=-2.5)[0][-1]
    k5 = np.where(etabins<=-1.3)[0][-1]
    k2 = np.where(etabins<=0)[0][-1]
    k3 = np.where(etabins<=1.3)[0][-1]
    
    ax.plot(ptbins[start:],mean_p[start:,k1], 'o', label=f'${etabins[k1]}<\eta<{etabins[k1+1]}$')
    ax.plot(ptbins[start:],mean_p[start:,k2], 'o', label=f'${etabins[k2]}<\eta<{etabins[k2+1]}$')
    ax.plot(ptbins[start:],mean_p[start:,k3], 'o', label=f'${etabins[k3]}<\eta<{etabins[k3+1]}$')
    
    ax.set_xlabel(r'$p_T$ (GeV)');
    ax.set_ylabel(r'mean response');
    ax.set_xscale('log')
    ax.legend()
    if test_run:
        plt.savefig('fig/corr_vs_pt'+samp+tag_full+'_test.pdf');
    else:
        plt.savefig('fig/corr_vs_pt'+samp+tag_full+'.pdf');
    
    plt.show();
    
    yvals
    
    np.max(yvals)
    
    np.where(yvals<np.median(yvals))
    
    rms = np.sqrt(np.sum((yvals-np.max(yvals))**2/len(yvals)))
    
    rms
    
    yvals
    
    np.mean(yvals)
    
    np.where(yvals<np.median(yvals)-rms)
    
    xfit_l
    xfit_h 
    
    yvals
    
    histo = output['ptresponse'].integrate('jeteta', hist.Interval(-2.5, -1.3)).integrate('pt', hist.Interval(90, 120))
    h = np.max(histo.values()[('QCD',)])
    h = h if h!=0 else 0.05
    xvals = output['ptresponse'+samp].axis('ptresponse').centers()
    yvals = histo.values()[('QCD',)]
    xvals = xvals[1:]
    yvals = yvals[1:]
    scal = scalings.loc[-2.5,str(300.0)]
    p, arr = curve_fit(gauss, xvals, yvals, p0=[10,1,1])
    
    fgaus = gauss(f_xvals, *p)
    
    histvals = np.repeat(histo.axis('ptresponse').centers(), np.array(histo.values()[('QCD',)],dtype='int'))
    
    median[i,k] = np.median(histvals)
    mean[i,k] = p[1]-(scal-1)
    width[i,k] = p[2]
    idx.append(i)
    
    ygaus = gauss(xvals, *p)
    chi2 = sum((yvals-ygaus)**2/(yvals+1E-9))
    N = histo.integrate('ptresponse').values()[('QCD',)]
    plot_response_axis = hist.Bin("jeteta", r"Jet $\eta$", hist_pt_edges)
    histo = histo.rebin('ptresponse', plot_response_axis)
    
    p1 = p
    
    p
    
    plt.plot((yvals-ygaus)**2/(yvals+1E-9))
    
    np.sqrt(arr[1,1])
    
    all# %%capture
    fig, ax2 = plt.subplots()
    hist.plot1d(histo, ax=ax2, overlay='dataset', overflow='all',
                fill_opts={'alpha': .5, 'edgecolor': (0,0,0,0.3), 'linewidth': 1.4})
    ax2.plot(f_xvals, fgaus, label='Gaus',linewidth=1.8)
    ax2.set_xlim(plot_pt_edges[[0,-1]])
    plt.text(1.4,0.75*h,'Mean {0:0.3f}'.format(p[1]))
    plt.text(1.4,0.68*h,'Median {0:0.3f}'.format(np.median(histvals)))
    plt.text(1.4,0.61*h,'Width {0:0.3f}'.format(p[2]))
    plt.text(1.3,0.53*h,'$\chi^2/ndof$ {0:0.1f}/97'.format(chi2))
    plt.text(1.3,0.46*h,'N data = {0:0.0f}'.format(N))
    ax2.legend(ncol=2);
    
    plt.show();
    plt.close();
    
    all# %%capture
    fig, ax2 = plt.subplots()
    hist.plot1d(histo, ax=ax2, overlay='dataset', overflow='all',
                fill_opts={'alpha': .5, 'edgecolor': (0,0,0,0.3), 'linewidth': 1.4})
    ax2.plot(f_xvals, fgaus, label='Gaus',linewidth=1.8)
    ax2.set_xlim(plot_pt_edges[[0,-1]])
    plt.text(1.4,0.75*h,'Mean {0:0.2f}'.format(p[1]))
    plt.text(1.4,0.68*h,'Median {0:0.2f}'.format(np.median(histvals)))
    plt.text(1.4,0.61*h,'Width {0:0.2f}'.format(p[2]))
    plt.text(1.3,0.53*h,'$\chi^2/ndof$ {0:0.1f}/97'.format(chi2))
    plt.text(1.3,0.46*h,'N data = {0:0.0f}'.format(N))
    ax2.legend();
    
    plt.show();
    plt.close();
    
    data = {str(ptBin):mean[i] for i, ptBin in enumerate(ptbins)}
    
    data['etaBins'] = np.array([str(etaBin) for etaBin in etabins])
    
    df = pd.DataFrame(data=data)
    df = df.set_index('etaBins')
    if not test_run:
        df.to_csv('out_txt/EtaBinsvsPtBinsMean'+samp+tag_full+'.csv')
    else:
        df.to_csv('out_txt/EtaBinsvsPtBinsMean'+tag+'_test.csv')
    
    data_width = {str(ptBin):width[i] for i, ptBin in enumerate(ptbins)}
    
    data_width['etaBins'] = [str(etaBin) for etaBin in etabins]
    
    df_width = pd.DataFrame(data=data_width)
    df_width = df_width.set_index('etaBins')
    if not test_run:
        df_width.to_csv('out_txt/EtaBinsvsPtBinsWidth'+samp+tag_full+'.csv')
    else:
        df_width.to_csv('out_txt/EtaBinsvsPtBinsWidth'+tag+'_test.csv')
    
    len(data['etaBins'])
    len(mean[0])
    
    data_median = {str(ptBin):median[i] for i, ptBin in enumerate(ptbins)}
    
    data_median['etaBins'] = [str(etaBin) for etaBin in etabins]
    
    df_median = pd.DataFrame(data=data_median)
    df_median = df_median.set_index('etaBins')
    if not test_run:
        df_median.to_csv('out_txt/EtaBinsvsPtBinsMedian'+samp+tag_full+'.csv')
    else:
        df_median.to_csv('out_txt/EtaBinsvsPtBinsMedian'+tag+'_test.csv')
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
        
    
    df_csv = pd.read_csv('out_txt/EtaBinsvsPtBinsMean_L5.csv').set_index('etaBins')
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df_csv)
    
if __name__ == "__main__":
    main()