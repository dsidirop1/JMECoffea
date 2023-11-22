#!/usr/bin/env python
    # coding: utf-8
### run_processor.py
"""
Submits a coffea processor on an executor like iterative, dask, condor or coffea-casa. 

Author(s): Andris Potrebko (RTU)
"""
# workaround to get a locally installed coffea and awkwrd version using lch on lxplus
# comment out or replace the path if I happened to forget to remove these lines before pushing:
import sys
import os
coffea_path = '/afs/cern.ch/user/a/anpotreb/top/JERC/coffea/'
if not os.path.exists(coffea_path):
    raise ValueError(f"The path to the coffea installation does not exist. Please supply the correct path or comment out this line if using the environment path. The provided path is: {coffea_path}.")
if coffea_path not in sys.path:
    sys.path.insert(0,coffea_path)
# ### Imports 
import os
import optparse
from packaging import version
import coffea
if version.parse(coffea.__version__) < version.parse('0.7.21'):
    raise ValueError(f"The coffea version used is {coffea.__version__} which is a buggy version when used on dask. Either update to 0.7.21 or remove this error statement.")

# import coffea
# print("Coffea version: ", coffea.__version__)

import time
from coffea import processor, util
from coffea.nanoevents import NanoAODSchema, BaseSchema
import datetime

# from numpy.random import RandomState
import importlib
from fileNames.available_datasets import dataset_dictionary
from helpers import get_xsecs_filelist_from_file, append_endpoint_redi


def run_processor(
        data_tag:str='',
        dataset:str=None,
        fileslist:list=None,
        add_tag:str='',
        run_comment:str='',
        executor:str='iterative',
        test_run:bool=True,
        Nfiles:int=-1,
        blacklist_sites:list = [], #'T2_IT_Rome'
        get_exact_endpoints:bool = False,
        xrootdstr:str = 'root://xrootd-cms.infn.it/',
        ):
    '''
    Submits a coffea processor on an executor like iterative, dask, condor or coffea-casa. 
    Inputs:
    - The dataset can be defined either by using a `data_tag` available in `dataset_dictionary`
    or manually by defining `dataset` (below) with the path to the .txt file with the file names
    or manually by defining `fileslist` as the list with file names.
    `data_tag` will be used to name output figures and histograms.
    - `add_tag` is the name of the specific run if parameters changed used for saving figures and output histograms.
    - `run_comment` is the comment for the log file: e.g., why the run is made?
    - `executor` can be 'iterative' for local (slow/concurrent) iterative executor; 'dask' for local (parallel) dask executor;
               'condor' for dask on condor; 'coffea-casa' for dask on coffea-casa
    - `test_run` is True if run only on one file and five chunks to debug processor (overrides Nfiles)
    - `Nfiles` is the number of files to run on. -1 for all files.
    - `blacklist_sites` is the list of sites to blacklist when/if obtaining exact endpoints of the files. e.g., ['T2_IT_Rome']
    - `get_exact_endpoints` is True if want to obtain the exact endpoints of the files. False if want to use the provided redirector.
    - `xrootdstr` is the redirector to use if `get_exact_endpoints` is False. default, 'root://xrootd-cms.infn.it/'

    '''
    ################ Parameters of the run and switches  #########################
    load_preexisting  = False    ### True if don't repeat the processing of files and use preexisting JER from output

    # Nfiles = -1 if ('Pythia-semilep-TTBAR' not in data_tag) or ('Pythia-TTBAR' not in data_tag) else 100                  ### number of files for each sample; -1 for all files
    # Nfiles = -1
    

    tag_Lx = '_L5'                 ### L5 or L23, but L23 not supported since ages.
    processor_name = 'CoffeaJERCProcessor'+tag_Lx
    from CoffeaJERCProcessor_L5_config import processor_config, processor_dependencies
    # processor_name = 'Processor_correctionlib_issue'
    # processor_config = None
    # processor_dependencies = []


    ### name of the specific run if parameters changed used for saving figures and output histograms.
    # add_tag = '' # _iso_dr_0p8 '_3rd_jet' # _cutpromtreco _Aut18binning   
    # run_comment = 'Testing the the new settings with '
    # run_comment = 'Testing the leading generated jets cut'
    # run_comment = 'Testing running the sample with scaled pions.'                          
    #### Comment for the log file: e.g., why the run is made?
    
    ### Define the dataset either by using a `data_tag` available in `dataset_dictionary`
    ### Or manually by defining `dataset` (below) with the path to the .txt file with the file names (without the redirectors).
    ### Or manually by defining `fileslist` as the list with file names.
    ### data_tag will be used to name output figures and histograms.

    log_basename = "/condor_coffea_log/condor_coffea_log_"
    # ### Specify datasets separatelly
    # dataset = None
    # fileslist = None
    # fileslist = ['root://xrootd-cms.infn.it//store/mc/RunIISummer20UL18NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/2520000/C9B528C2-F9A0-5D48-851B-534BCA92532F.root']

    ### Choose the correct redirector
    ## assume running on the LPC
    # xrootdstr = 'root://cmsxrootd.fnal.gov/'
    ## assume running on the lxplus
    # xrootdstr = 'root://cms-xrd-global.cern.ch//'
    # xrootdstr = 'root://xrootd-cms.infn.it/' #if 'TTBAR' not in data_tag else ''
    # xrootdstr = ''

    ################ End of Parameters of the run and switches  #########################
    
    # ### Obtain the dataset and cross-sections from the dataset_dictionary. Define and print the information about the run.
    printout = f'Processor {processor_name} will be run on {executor}.\n'  ### to be used in the saved output to .txt
    tag_full = tag_Lx+'_'+data_tag+add_tag

    file_setup_is_none = [fileslist is None, dataset is None, data_tag == '']
    if sum(file_setup_is_none)==3:
        raise ValueError(f'No data tag, dataset or fileslist provided. Please provide either data_tag, dataset or fileslist.')
    if sum(file_setup_is_none)<2:
        raise ValueError(f'Please provide either data_tag, dataset or fileslist. Not more than one of these can be provided.')

    if not (fileslist is None):
        xsec = 1
        legend_label = add_tag
        printout += f'A specific fileslist specified. The calculation will be run on the files:\n{fileslist}.\nThe histograms will be saved with the provided tag {add_tag} \n'
    elif not(dataset is None):
        xsec = 1
        legend_label = add_tag
        printout += (f'Using the provided dataset with the path to file names {dataset[0]}.\nThe histograms will be saved with the provided tag {add_tag} \n')
    elif data_tag in dataset_dictionary.keys():
        dataset, xsec, legend_label = dataset_dictionary[data_tag]
        if dataset is None:
            printout += (f'The data tag "{data_tag}" found in the dataset_dictionary. The dataset with the path to cross-sections {xsec} will be used. \n')
        else:
            printout += f'The data tag "{data_tag}" found in the dataset_dictionary. The dataset with the path to file names "{dataset}" will be used. \n'
    else:
        raise ValueError(f'The data tag "{data_tag}" not found in the dataset_dictionary and no dataset provided.')
    print(printout)
    
    # if running on coffea casa instead...
    if executor=='coffea-casa':
        print("Running on coffea casa. Changing the xrootdstr to 'root://xcache/'")
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
    
    chunksize = 30000 #25000 
    ## hard to adjust the chunksize correctly. Without LHE flavor matching 50 000/ less for hadronic is usually fine, with LHE flavor, needs to decrease to 25k.  
    print("chunksize used = ", chunksize)
    
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    username = os.environ['USER']
    condor_log_dir = '/eos/home-'+username[0]+'/'+username+log_basename+suffix

    printout_tmp = f'Running on the number of files: {Nfiles}\n Job with the full tag {tag_full}\n Outname = {outname} \n'
    printout_tmp += f'condor log will be saved under {condor_log_dir}'     if executor == 'condor' else '' 
    print(printout_tmp)
    printout += printout_tmp
    
    def txt2filesls(dataset_name, Nfiles=Nfiles):
        with open(dataset_name) as f:
            rootfiles = f.read().split()
            if Nfiles==-1:
                Nfiles = len(rootfiles)
            rootfiles = rootfiles[:Nfiles]
            has_xrootd = 'root://' in rootfiles[0]
            prepend_str = '' if has_xrootd else xrootdstr
            if get_exact_endpoints and not has_xrootd:
            # if False:
                out_name = dataset_name.split('.')[0]+'_redi.txt'
                fileslist = append_endpoint_redi(rootfiles, out_name, blacklist_sites)
            else:
                fileslist = [prepend_str + file for file in rootfiles]
        return fileslist
    
    if fileslist is not None:
        xsec_dict = {data_tag: xsec}
        fileslist2 = []
        for file in fileslist:
            if 'root://' in file:
                fileslist2.append(file)
            else:
                fileslist2.append(xrootdstr+file)
        fileslist = fileslist2
        filesets = {data_tag: {"files": fileslist, "metadata": {"xsec": xsec}}}
    elif (dataset is None) and (xsec is not None):
        ### if dataset striched together from a set of datasets where the cross-section for each is given in `xsec`
        xsec_dict, file_dict = get_xsecs_filelist_from_file(xsec, data_tag)
        path_to_xsec = '/'.join(xsec.split('/')[:-1])
        filesets = {}
        for key in file_dict.keys():
            data_name = file_dict[key]
            fileslist = txt2filesls(path_to_xsec+'/'+data_name, Nfiles)
            filesets[key] = {"files": fileslist, "metadata": {"xsec": xsec_dict[key]}}
    else:
        fileslist = txt2filesls(dataset, Nfiles)
        #### If manyally adding fileslist
        xsec_dict = {data_tag: xsec}
        filesets = {data_tag: {"files": fileslist, "metadata": {"xsec": xsec}}}

    if not load_preexisting:
        import uproot
        print(f'\nTesting to open one file \n{fileslist[0]}\nThis will fail if certificates are not set up or the filenames are wrong or just fail because uproot has a bad day.')
        ff = uproot.open(fileslist[0])
        ff.keys()
        ff.close()
        print(f"The test file read successfully. All good with the certifiates.\n")
    # breakpoint()

    ########### Initiate the client and set up the executors ###########
    if(executor == 'coffea-casa'):
        # Dask set up for Coffea-Casa only
       from dask.distributed import Client 
       client = Client("tls://ac-2emalik-2ewilliams-40cern-2ech.dask.coffea.casa:8786")
       client.upload_file(processor_name+'.py')
       for dep in processor_dependencies:
          client.upload_file(dep)
    elif(executor=='condor' or executor=='dask'):
        from dask.distributed import Client 
        if executor=='dask':
            client = Client()
            client.get_versions(check=True)
    #         client.nanny = False
        elif executor=='condor':
            from dask_lxplus import CernCluster
            import socket
# 
            # args = {"mild_scaleout": True, "scaleout": 2, "max_scaleout": 600}
            # if args["mild_scaleout"]:
            #     adapt_parameters = dict(
            #         interval="1m",
            #         target_duration="30s",
            #         wait_count=10,
            #     )
            # else:
            #     adapt_parameters = dict()
    
            cluster = CernCluster(
                cores = 1,
                memory = '4000MB',
                disk = '2000MB',
                death_timeout = '60',
                lcg = True,
                nanny = True,
                container_runtime = 'none',
                log_directory = condor_log_dir,
                scheduler_options = {
                    'port': 8786,
                    'host': socket.gethostname(),
                },
                job_extra = {
                    'MY.JobFlavour': '"espresso"',
                    'MY.AccountingGroup': '"group_u_CMST3.all"',
                },
            )
            cluster.adapt(minimum=2, maximum=50)
            # cluster.adapt(
            #     minimum=args["scaleout"],
            #     maximum=args["max_scaleout"],
            #     **adapt_parameters,
            # )
            # cluster.scale(10)
            client = Client(cluster)
        client.upload_file(processor_name+'.py')
        for dep in processor_dependencies:
            client.upload_file(dep)

        # print("Printing the client information: \n", client)
        print("Printing the client information:\nFor cern cluster this shows processes=0 etc which is fine\n", client)
        # breakpoint()
    elif executor=='iterative' or executor=='futures':
        print(f"Running on {executor} executor.")
    else:
        ValueError(f"Executor {executor} not supported. Please choose 'iterative', 'dask', 'condor' or 'coffea-casa'.")
    ########### End of the Initiate the client and set up the executors ###########

    # breakpoint()
    # ### Run the processor
    
    tstart = time.time()
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
                                                  'treereduction':5,
    #                                               'workers': 2
                                              },
                                              chunksize=chunksize,
                                              maxchunks=maxchunks)
    
        elapsed = time.time() - tstart
        print(f"Processor finished. Time elapsed: {elapsed//60} min. {elapsed%60} sec.")
        print("Saving the output histograms under: ", outname)
        util.save(output, outname)
    else:
        output = util.load(outname)
        print("Loaded histograms from: ", outname)
    
    #### Attempt to prevent the error when the cluster closes. Doesn't always work.
    if executor=='condor' or executor=='coffea-casa':
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
            log_file.writelines(['\n' + printout, '\nConfig parameters:\n' + str(processor_config)+ '\n'])
    
    print('-----'*10)
    print("All done. Congrats!")

def main():
    #configuration
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-d', '--data',        dest='data_tag',    help='data tag from the available datasets',      default=None,        type='string')
    (opt, args) = parser.parse_args()
    data_tags = [opt.data_tag]
    # data_tags = ['Pythia-non-semilep-TTBAR'] if opt.data_tag is None else [data_tags] # [, 'DY-MG-Her', 'QCD-MG-Her', 'Pythia-TTBAR', 'Herwig-TTBAR']
    data_tags = ['QCD-MG-Her'] if opt.data_tag is None else [data_tags] # ['Pythia-TTBAR', 'Herwig-TTBAR']
    # data_tags = ['Pythia-TTBAR'] if opt.data_tag is None else [data_tags] # ['Pythia-TTBAR', 'Herwig-TTBAR']
    params =      {"run_comment": 'Reruning ttbar all decay ch. for only 10 files. condor strugles to give more jobs.',
                #   "blacklist_sites":['T2_IT_Rome'],
                  "get_exact_endpoints":False,
                  "add_tag":'',
                  "Nfiles": -1,
                  
                  } 
    # for data_tag in data_tags:
        # run_processor(data_tag=data_tag, test_run=False, executor='condor', **params)
    # run_processor(fileslist=[
    #                         'root://osg-se.sprace.org.br:1094//store/mc/RunIISummer20UL18NanoAODv9/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/250000/A1EF1097-A3D4-1544-BB95-806AE84BB83E.root',
    #                         'root://xrootd-cms.infn.it//store/mc/RunIISummer20UL18NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/230000/9D0C102A-1A88-7D48-80A7-509AB9EAFD26.root',
    #                         'root://osg-se.sprace.org.br:1094//store/mc/RunIISummer20UL18NanoAODv9/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/2520000/2A1DBFB7-B746-9D4A-B017-E8221D4AEA6D.root',],
    #                         executor='iterative',
    #                         test_run=False,
    #                         **params)
    run_processor(fileslist=[
                            '/store/mc/RunIISummer20UL18NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/30000/4988713D-E70D-E243-A384-B902119A3604.root',
                            '/store/mc/RunIISummer20UL18NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/30000/519DE155-138B-DE46-92CC-6460F9172458.root',
                            '/store/mc/RunIISummer20UL18NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/30000/59ECE256-116E-E042-BA04-E415FCDA1A3B.root',],
                            executor='iterative',
                            test_run=False,
                            **params)

if __name__ == "__main__":
    main()
