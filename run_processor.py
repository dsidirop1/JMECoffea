#!/usr/bin/env python
    # coding: utf-8
### run_processor.py

# ### Imports 
#### Import updated coffea and awkward versions    
import sys
coffea_path = '/afs/cern.ch/user/a/anpotreb/top/JERC/coffea/'
if coffea_path not in sys.path:
    sys.path.insert(0,coffea_path)

ak_path = '/afs/cern.ch/user/a/anpotreb/top/JERC/local-packages/'

if ak_path not in sys.path:
    sys.path.insert(0,ak_path)

import time
from coffea import processor, util
from coffea.nanoevents import NanoAODSchema, BaseSchema

from numpy.random import RandomState
import importlib
import os
from fileNames.available_datasets import dataset_dictionary
from helpers import get_xsecs_filelist_from_file


def main():
    
    #### The readme has to be rewritten!!!

    # The file is a wrapper for running coffea processor defined under `processor_name` over jobs with NanoAOD files with working futures, dask and condor settings. 
    # The default processor `CoffeaJERCProcessor_L5.py`, the histograms of jet reponses and reco jet $p_T$. Stores the result in outname='out/CoffeaJERCOutputs'tag_Lx+'_'+data_tag+add_tag'.coffea' for the default processor or in 'out/'+processor_name+tag_full+'.coffea' for the other processors.
    # 
    # Options for running on condor, coffea Case using dask are available. See, under `Parameters of the run and switches`.
    # 
    # At the moment, since dask+condor can still sometimes be unstable, for producing the flavor uncertatinties, where one needs to obtain the results from neccessary 6 datasets neccessary, the script has to be run once for each of them (QCD stiching is done automatically). However, this can and should be united in the future.
    # The available datasets are listed in `fileNames/available_datasets.py`. The dataset for the run can be chosen with a corresponding `data_tag` or by providing a path to the .txt file with the file names (without the redirectors) in `dataset`. For small local testing dataset can be also specified as a list of files in `fileslist`.
    
    # # Dask Setup:
    # ---
    # ### For Dask+Condor setup on lxplus
    # #### 1.) The wrapper needs to be installed following https://github.com/cernops/dask-lxplus
    # #### 2.) Source lcg environment in bash
    # #### `source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4/latest/x86_64-centos7-gcc11-opt/setup.sh`
    # #### Singularity could work but not confirmed.
    # ---
    # ### For Coffea-Casa, the client must be specified according to the user that is logged into the Coffea-Casa Environment.
    # #### 1.) go to the left of this coffea-casa session to the task bar and click the orange-red button; it will say "Dask" if you hover your cursor over it
    # #### 2.) scroll down to the blue box where it shows the "Scheduler Address"
    # #### 3.) write that full address into the dask Client function 
    # #### Example: `client = Client("tls://ac-2emalik-2ewilliams-40cern-2ech.dask.coffea.casa:8786")`
    # ---
    # ### For CMSLPC, the client must be specified with the LPCCondorCluster
    # #### 1.) follow installation instructions from https://github.com/CoffeaTeam/lpcjobqueue, if you have not already done so, to get a working singularity environment with access to lpcjobqueue and LPCCondorCluster class
    # #### 2.) import LPCCondorCluster: `from lpcjobqueue import LPCCondorCluster`
    # #### 3.) define the client
    # #### Example: 
    # `cluster = LPCCondorCluster()`
    # 
    # `client = Client(cluster)`
    # 

    ################ Parameters of the run and switches  #########################
    
    # 'iterative' for local (slow/concurrent) iterative executor; 'dask' for local (parallel) dask executor;
    # 'condor' for dask on condor; 'coffea-casa' for dask on coffea-casa
    executor = 'condor' 
    load_preexisting  = False    ### True if don't repeat the processing of files and use preexisting JER from output
    test_run          = False   ### True if run only on one file and five chunks to debug processor

    Nfiles = -1                 ### number of files for each sample; -1 for all files
    
    tag_Lx = '_L5'                 ### L5 or L23, but L23 not supported since ages.
    processor_name = 'CoffeaJERCProcessor'+tag_Lx
    from CoffeaJERCProcessor_L5_config import processor_config, processor_dependencies
    # processor_name = 'Processor_HT_spectrum'
    # processor_config = None
    # processor_dependencies = []


    
    ### Define the dataset either by using a `data_tag` available in `dataset_dictionary`
    ### Or manually by defining `dataset` (below) with the path to the .txt file with the file names (without the redirectors).
    ### Or manually by defining `fileslist` as the list with file names.
    ### data_tag will be used to name output figures and histograms.
    data_tag = 'Herwig-TTBAR' # 'QCD-MG-Her' #'Herwig-TTBAR' 
    # data_tag = 'DY-FxFx'
    ### name of the specific run if parameters changed used for saving figures and output histograms.
    add_tag = '_iso_cut' #'_3rd_jet' # _cutpromtreco _Aut18binning   
    run_comment = 'Rerunning the isolation cut run with all the statistics'                          #### Comment for the log file: e.g., why the run is made?
    
    certificate_dir = '/afs/cern.ch/user/a/anpotreb/k5-ca-proxy.pem'

    import datetime
    log_basename = "/condor_coffea_log/condor_coffea_log_"
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # tmpdir = os.environ['TMPDIR']+log_basename+suffix
    # tmpdir = os.environ['HOME']+'/log/'+log_basename+suffix
    # condor_log_dir = tmpdir
    username = os.environ['USER']
    condor_log_dir = '/eos/home-'+username[0]+'/'+username+log_basename+suffix
    # ### Specify datasets separatelly
    dataset = None
    fileslist = None
    # fileslist = ['root://xrootd-cms.infn.it/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/2510000/056516']

    ### Choose the correct redirector
    ## assume running on the LPC
    # xrootdstr = 'root://cmsxrootd.fnal.gov/'
    ## assume running on the lxplus
    # xrootdstr = 'root://cms-xrd-global.cern.ch//'
    xrootdstr = 'root://xrootd-cms.infn.it/'

    ################ End of Parameters of the run and switches  #########################
    
    # ### Obtain the dataset and cross-sections from the dataset_dictionary. Define and print the information about the run.
    printout = f'Processor {processor_name} will be run on {executor}.\n'  ### to be used in the saved output to .txt
    tag_full = tag_Lx+'_'+data_tag+add_tag
    if not (fileslist is None):
        xsec = 1
        legend_label = data_tag
        printout += f'A specific fileslist specified. The calculation will be run on the files {fileslist}. The histograms will be saved with the provided tag {data_tag} \n'
    elif data_tag in dataset_dictionary.keys():
        dataset, xsec, legend_label = dataset_dictionary[data_tag]
        if dataset is None:
            printout += (f'The data tag "{data_tag}" found in the dataset_dictionary. The dataset with the path to cross-sections {xsec} will be used. \n')
        else:
            printout += f'The data tag "{data_tag}" found in the dataset_dictionary. The dataset with the path to file names "{dataset}" will be used. \n'
    elif not(dataset is None):
        xsec = 1
        legend_label = data_tag
        printout += (f'Using the provided dataset with the path to file names {dataset[0]} and the provided data tag "{data_tag}". \n')
    else:
        raise ValueError(f'The data tag "{data_tag}" not found in the dataset_dictionary and no dataset provided.')
    print(printout)
    
    # if running on coffea casa instead...
    if executor=='coffea-casa':
        xrootdstr = 'root://xcache/'
    
    # ### Do some logic with the input partameters and the rest of parameters of the run
    
    #Import the correct processor
    Processor = importlib.import_module(processor_name).Processor    
    tag_full = tag_Lx+'_'+data_tag+add_tag
    if test_run:
        tag_full = tag_full+'_test'

    if processor_name=='CoffeaJERCProcessor_L5':
        outname = 'out/CoffeaJERCOutputs'+tag_full+'.coffea'
    else:
        outname = 'out/'+processor_name+tag_full+'.coffea'
        
    if not os.path.exists("out"):
        os.mkdir("out")
        
    maxchunks = 5 if test_run else None
    if test_run:
        Nfiles = 1
    
    printout_tmp = f'Running on the number of files: {Nfiles}\n Job with the full tag {tag_full}\n Outname = {outname} \n'
    printout_tmp += f'condor log will be saved under {condor_log_dir}'     if executor == 'condor' else '' 
    print(printout_tmp)
    printout += printout_tmp
    
    def txt2filesls(dataset_name):
        with open(dataset_name) as f:
            rootfiles = f.read().split()
            fileslist = [xrootdstr + file for file in rootfiles]
        return fileslist
    
    if fileslist is not None:
        xsec_dict = {data_tag: xsec}
        filesets = {data_tag: {"files": fileslist, "metadata": {"xsec": xsec}}}
    elif (dataset is None) and (xsec is not None):
        ### if dataset striched together from a set of datasets where the cross-section for each is given in `xsec`
        xsec_dict, file_dict = get_xsecs_filelist_from_file(xsec, data_tag, test_run)
        path_to_xsec = '/'.join(xsec.split('/')[:-1])
        filesets = {}
        for key in file_dict.keys():
            data_name = file_dict[key]
            fileslist = txt2filesls(path_to_xsec+'/'+data_name)[:Nfiles]
            filesets[key] = {"files": fileslist, "metadata": {"xsec": xsec_dict[key]}}
    else:
        fileslist = txt2filesls(dataset)[:Nfiles]
        #### If manyally adding fileslist
        xsec_dict = {data_tag: xsec}
        filesets = {data_tag: {"files": fileslist, "metadata": {"xsec": xsec}}}
    
    if not load_preexisting:
        import uproot
        ff = uproot.open(fileslist[0])
        ff.keys()
        ff.close()
        # print(f"The test file read successfully. All good with the certifiates.")
    
    # Dask set up for Coffea-Casa only
    if(executor == 'coffea-casa'):
       from dask.distributed import Client 
       client = Client("tls://ac-2emalik-2ewilliams-40cern-2ech.dask.coffea.casa:8786")
       client.upload_file(processor_name+'.py')
       for dep in processor_dependencies:
          client.upload_file(dep)
    
    if(executor=='condor' or executor=='dask'):
        from dask.distributed import Client 
        if executor=='dask':
            client = Client()
            client.get_versions(check=True)
    #         client.nanny = False
        elif executor=='condor':
            from dask_lxplus import CernCluster
            import socket
    
            cluster = CernCluster(
                cores = 1,
                memory = '4000MB',
                disk = '2000MB',
                death_timeout = '60',
                lcg = True,
                nanny = False,
                container_runtime = 'none',
                log_directory = condor_log_dir,
                scheduler_options = {
                    'port': 8786,
                    'host': socket.gethostname(),
                },
                job_extra = {
                    'MY.JobFlavour': '"longlunch"',
                },
            )
            cluster.adapt(minimum=2, maximum=200)
            cluster.scale(8)
            client = Client(cluster)
        
        client.upload_file(processor_name+'.py')
        for dep in processor_dependencies:
            client.upload_file(dep)

        print("Printing the client information: \n", client)
        
    
    # ### Run the processor
    
    tstart = time.time()
    
    seed = 1234577890
    prng = RandomState(seed)
    chunksize = 1000
    # maxchunks = 100
    
    if not load_preexisting:
        if executor=='iterative':
            output = processor.run_uproot_job(filesets,
                                              treename='Events',
                                              processor_instance=Processor(processor_config),
                                              executor=processor.iterative_executor,
                                              executor_args={
                                                  'skipbadfiles':True,
                                                  'schema': NanoAODSchema, #BaseSchema
                                                  'workers': 2},
                                              chunksize=chunksize,
                                              maxchunks=maxchunks)
        else:
            # chosen_exec = 'dask'
            output = processor.run_uproot_job(filesets,
                                              treename='Events',
                                              processor_instance=Processor(processor_config),
                                              executor=processor.dask_executor,
                                              executor_args={
                                                  'client': client,
                                                  'skipbadfiles':True,
                                                  'schema': NanoAODSchema, #BaseSchema
                                                  'xrootdtimeout': 60,
                                                  'retries': 2,
    #                                               'workers': 2
                                              },
                                              chunksize=chunksize,
                                              maxchunks=maxchunks)
    
        elapsed = time.time() - tstart
        print("Processor finished. Time elapsed: ", elapsed)
    #     outputs_unweighted[name] = output
        print("Saving the output histograms under: ", outname)
        util.save(output, outname)
    #     outputs_unweighted[name] = output
    else:
        output = util.load(outname)
        print("Loaded histograms from: ", outname)
    
    #### Attempt to prevent the error when the cluster closes. Doesn't always work.
    if executor=='condor' or executor=='coffea-casa' or executor=='dask':
        client.close()
        time.sleep(5)
        cluster.close()
    
    from helpers import find_result_file_index
    run_log_name = "run_log.txt"
    
    if not load_preexisting:
        run_idx = find_result_file_index(run_log_name)
        file_name_title = f'Run_index_{run_idx}'
        # log_file = run_log_name.open('a')
        with open(run_log_name, 'a') as log_file:
            log_file.writelines(['\n' + file_name_title + '\nRun comment: ' + run_comment])
            log_file.writelines(['\n' + printout, '\nConfig parameters:\n' + str(processor_config)])
    
    print('-----'*10)
    print("All done. Congrats!")

if __name__ == "__main__":
    main()
