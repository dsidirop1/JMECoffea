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
    test_run          = False     ### True if run only on one file
    
    tag = '_L5'
    
    outname = 'out/CoffeaJERCOutputs'+tag+'.coffea'
    outname = outname+'_test' if test_run else outname
    
    xrootdstr = 'root://cms-xrd-global.cern.ch//'
    
    if CoffeaCasaEnv:
        xrootdstr = 'root://xcache/'
    
    rootfiles = open('dataset.txt').read().split()
    
    fileslist = [xrootdstr + file for file in rootfiles]
    
    fileslist = fileslist
    if test_run:
        fileslist = [fileslist[0]]
        ### The smallest file in the RunIISummer20UL18NanoAODv9 dataset
        fileslist = ['root://cms-xrd-global.cern.ch//'+
                     '/store/mc/RunIISummer20UL18NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/30000/792B4CD3-A001-F94F-9AAB-D74D532DE610.root']
    
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
    
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
    ptbins = output['ptresponse'].axis('pt').edges()
    ptbins;
    etabins = output['ptresponse'].axis('jeteta').edges()[:-1]
    etabins;
    
    jetpt_length = len(ptbins)
    jeteta_length = len(etabins)
    
    mean = np.zeros((jetpt_length, jeteta_length))
    median = np.zeros((jetpt_length, jeteta_length))
    width = np.zeros((jetpt_length, jeteta_length))
    idx = []
    
    xvals = output['ptresponse'].axis('ptresponse').centers()
    f_xvals = np.linspace(0,5,5001)
    
    from pltStyle import pltStyle
    pltStyle(style='paper')
    
    output
    
    samp = '_b'
    
    import warnings
    warnings.filterwarnings('ignore') ### To suppress warnings with bad 
    
    N_converge = 0
    N_not_converge = 0
    
    xvals = output['ptresponse'+samp].axis('ptresponse').centers()
    f_xvals = np.linspace(0,5,5001)
    
    FitFigDir = 'fig/response_pt_eta'+samp+tag
    if not os.path.exists(FitFigDir):
        os.mkdir(FitFigDir)
    
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
            
    
            eta_string = '_eta'+str(etaBin.lo)+'to'+str(etaBin.hi)
            eta_string = eta_string.replace('.','')
       
            # The name integrate is a bit misleasding in this line. Is there another way to "slice" a histogram? //Andris
            histo = output['ptresponse'+samp].integrate('jeteta', etaBin).integrate('pt', ptBin) 
            
            histvals = np.repeat(histo.axis('ptresponse').centers(), np.array(histo.values()[('QCD',)],dtype='int'))
    
            yvals = histo.values()[('QCD',)]
            
            print("etaBin = ", etaBin ,", ptBin = ", ptBin )
            
            try:
                p, arr = curve_fit(gauss, xvals, yvals, p0=[10,1,1])
                print("Fit succeeded, p = ", p )
                N_converge += 1
            except:
                print("Fit failed, p = ", p)
                N_not_converge += 1
                continue
    
            
            fgaus = gauss(f_xvals, *p)
            
            median[i,k] = np.median(histvals)
            mean[i,k] = p[1]
            width[i,k] = p[2]
            idx.append(i)
            
    
            if not test_run:
                h = np.max(histo.values()[('QCD',)]);
                fig, ax2 = plt.subplots();
                hist.plot1d(histo, ax=ax2, overlay='dataset');
                ax2.plot(f_xvals, fgaus, label='Gaus');
                plt.text(3.8,0.75*h,'Mean {0:0.2f}'.format(p[1]));
                plt.text(3.8,0.68*h,'Median {0:0.2f}'.format(np.median(histvals)));
                plt.text(3.8,0.61*h,'Width {0:0.2f}'.format(p[2]));
                ax2.legend();
    
                # plt.show();
                plt.savefig(FitFigDir+'/ptResponse'+pt_string+eta_string+'.png');
                plt.close();        
            
            
    warnings.filterwarnings('default')
    print("N converge = ", N_converge, "N_not_converge = ", N_not_converge );    
    
    negmean = np.where((mean<0))
    negmean;
    ptbins[negmean[0]];
    etabins[negmean[1]];
    
    mean_p = mean.copy()
    mean_p[mean_p==0] = np.nan
    
    fig, ax = plt.subplots()
    start = 17
    k1 = 2
    k2 = np.where(etabins<=0)[0][-1]
    k3 = np.where(etabins<=2.5)[0][-1]
    ax.plot(ptbins[start:],mean_p[start:,k1], 'o', label=f'${etabins[k1]}<\eta<{etabins[k1+1]}$')
    ax.plot(ptbins[start:],mean_p[start:,k2], 'o', label=f'${etabins[k2]}<\eta<{etabins[k2+1]}$')
    ax.plot(ptbins[start:],mean_p[start:,k3], 'o', label=f'${etabins[k3]}<\eta<{etabins[k3+1]}$')
    ax.set_xlabel(r'$p_T$ (GeV)');
    ax.set_ylabel(r'mean response');
    ax.set_xscale('log')
    ax.legend()
    if test_run:
        plt.savefig('fig/corr_vs_pt'+samp+tag+'_test.png')
    else:
        plt.savefig('fig/corr_vs_pt'+samp+tag+'.png')
    
    plt.show();
    
    histo = output['ptresponse'].integrate('jeteta', hist.Interval(0, 0.087)).integrate('pt', hist.Interval(6, 7))
    
    try:
        p, arr = curve_fit(gauss, xvals, yvals, p0=[10,1,1])
        print("Fit succeeded, p = ", p)
    except:
        print("Fit failed, p = ", p)
    
    fgaus = gauss(f_xvals, *p)
    
    histvals = np.repeat(histo.axis('ptresponse').centers(), np.array(histo.values()[('QCD',)],dtype='int'))
    yvals = histo.values()[('QCD',)]
    
    median[i,k] = np.median(histvals)
    mean[i,k] = p[1]
    width[i,k] = p[2]
    idx.append(i)
    
    h = np.max(histo.values()[('QCD',)])
    fig, ax2 = plt.subplots()
    hist.plot1d(histo, ax=ax2, overlay='dataset')
    ax2.plot(f_xvals, fgaus, label='Gaus')
    plt.text(3.8,0.75*h,'Mean {0:0.2f}'.format(p[1]))
    plt.text(3.8,0.68*h,'Median {0:0.2f}'.format(np.median(histvals)))
    plt.text(3.8,0.61*h,'Width {0:0.2f}'.format(p[2]))
    ax2.legend()
    
    plt.close();
    
    etabins
    
    data = {str(ptBin):mean[i] for i, ptBin in enumerate(ptbins)}
    
    data['etaBins'] = np.array([str(etaBin) for etaBin in etabins])
    
    df = pd.DataFrame(data=data)
    df = df.set_index('etaBins')
    if test_run:
        df.to_csv('out_txt/EtaBinsvsPtBinsMean'+samp+tag+'.csv')
    
    data_width = {str(ptBin):width[i] for i, ptBin in enumerate(ptbins)}
    
    data_width['etaBins'] = [str(etaBin) for etaBin in etabins]
    
    df_width = pd.DataFrame(data=data_width)
    df_width = df_width.set_index('etaBins')
    if test_run:
        df_width.to_csv('out_txt/EtaBinsvsPtBinsWidth'+samp+tag+'.csv')
    
    len(data['etaBins'])
    len(mean[0])
    
    data_median = {str(ptBin):median[i] for i, ptBin in enumerate(ptbins)}
    
    data_median['etaBins'] = [str(etaBin) for etaBin in etabins]
    
    df_median = pd.DataFrame(data=data_median)
    df_median = df_median.set_index('etaBins')
    if test_run:
        df_median.to_csv('out_txt/EtaBinsvsPtBinsMedian'+samp+tag+'.csv')
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
        
    
if __name__ == "__main__":
    main()