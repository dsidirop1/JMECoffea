#CoffeaJERCProcessor_L5.py
''' coffea processor for calculating the jet energy response in bins of pt_ptcl and jet_eta.
    The processor makes a separate histogram for each jet flavor.
output: a dictionary over datasets of dictionaries over histograms.
output histograms: ptresponse histogram, pt_reco histogram for each flavor and the cutflow
''' 

# from memory_profiler import profile
from common_binning import JERC_Constants
from JERCProcessorcuts import jet_iso_cut, leading_jet_and_alpha_cut, jet_pt_cut, good_lepton_cut, select_leptons, recolep_drcut, select_Nth_jet, jetMCmatching, remove_apply

# workaround to get a locally installed coffea and awkwrd version using lch on lxplus
# comment out or replace the path if I happened to forget to remove these lines before pushing:
# import sys
# import os
# coffea_path = '/afs/cern.ch/user/a/anpotreb/top/JERC/coffea/'
# if not os.path.exists(coffea_path):
#     raise ValueError(f"The path to the coffea installation does not exist. Please supply the correct path or comment out this line if using the environment path. The provided path is: {coffea_path}.")
# if coffea_path not in sys.path:
#     sys.path.insert(0,coffea_path)
# 
# ak_path = '/afs/cern.ch/user/a/anpotreb/top/JERC/local-packages/'
# if ak_path not in sys.path:
#         sys.path.insert(0,ak_path)
# sys.path.insert(0,'/afs/cern.ch/user/a/anpotreb/top/JERC/JMECoffea')
# print("sys path = ", sys.path)
# from os import listdir
# listdir('.')
# listdir('./coffea')

from coffea import processor
import numpy as np
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
import correctionlib

from LHE_flavour import get_LHE_flavour, get_LHE_flavour_2
import hist
import awkward as ak

# from coffea import some_test_func
# some_test_func.test_func()

# manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]
ptbins = np.array(JERC_Constants.ptBinsEdgesMCTruth())
etabins = np.array(JERC_Constants.etaBinsEdges_CaloTowers_full())

class Processor(processor.ProcessorABC):
    def __init__(self, processor_config):   
        self.cfg = processor_config
        self.jetflavour = processor_config["jetflavour"]

        ext = extractor()
        ext.add_weight_sets([
            "* * Summer20UL18_V2_MC/Summer20UL18_V2_MC_L1FastJet_AK4PFchs.txt",
            "* * Summer20UL18_V2_MC/Summer20UL18_V2_MC_L2Relative_AK4PFchs.txt",
            "* * Summer20UL18_V2_MC/Summer20UL18_V2_MC_L3Absolute_AK4PFchs.txt",
#             "* * Summer20UL18_V2_MC/Summer19UL18_V5_MC_L2L3Residual_AK4PFchs.txt", #Doesn't do anything but for transparancy I add it
        ])
        ext.finalize()

        jec_stack_names = ["Summer20UL18_V2_MC_L1FastJet_AK4PFchs",
                           "Summer20UL18_V2_MC_L2Relative_AK4PFchs", 
                           "Summer20UL18_V2_MC_L3Absolute_AK4PFchs",
#                            "Summer19UL18_V5_MC_L2L3Residual_AK4PFchs",
                          ]

        evaluator = ext.make_evaluator()        
        jec_inputs = {name: evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)

        self.name_map = jec_stack.blank_name_map
        self.name_map['JetPt'] = 'pt'
        self.name_map['JetMass'] = 'mass'
        self.name_map['JetEta'] = 'eta'
        self.name_map['JetA'] = 'area'
        self.name_map['ptGenJet'] = 'pt_gen'
        self.name_map['ptRaw'] = 'pt_raw'
        self.name_map['massRaw'] = 'mass_raw'
        self.name_map['Rho'] = 'rho'
        
        
        self.jet_factory = CorrectedJetsFactory(self.name_map, jec_stack)
        self.flavor2partonNr = {'b':5,
                           'c':4,
                           's':3,
                           'u':2,
                           'd':1,
                           'bbar':-5,
                           'cbar':-4,
                           'sbar':-3,
                           'ubar':-2,
                           'dbar':-1,
                           'g':21,
                           'unmatched':0,
                           }
        
        self.flavors = self.flavor2partonNr.keys() #['b', 'c', 'u', 'd', 's', 'g', 'bbar', 'cbar', 'ubar', 'dbar', 'sbar', 'untagged']

        path_to_PU_weights = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2018_UL/puWeights.json.gz"
        self.pucorr = correctionlib.CorrectionSet.from_file(path_to_PU_weights)

#         ### Attempt to apply closure from a prederived file
#        df_csv = pd.read_csv('out_txt/Closure_L5_QCD_Pythia.coffea').set_index('etaBins')
#        self.closure_corr = df_csv.to_numpy().transpose()
#        self.closure_corr = np.pad(self.closure_corr,1,constant_values=1)
#        self.etabins_closure = df_csv.index.to_numpy()
#        self.ptbins_closure = df_csv.columns.to_numpy('float')
        
#         ### Uncomment to set closure as 1
#         self.closure_corr = np.ones(self.closure_corr.shape)
#     print(f"awkward version {ak.__version__}")
            
    @property
    def accumulator(self):
        return self._accumulator

    # @profile    
    # def for_memory_testing(self):
    #     a=1
        
#     @profile    
    def process(self, events):

        ############ Define the histograms ############
        flavors = self.flavors        
        # flavour_axis = hist.axis.StrCategory(flavors, growth=False, name="jet_flav", label=r"jet_flavour")  ###not completelly sure if defining an axis is better than doing through a dictionary of flavors. See, https://github.com/CoffeaTeam/coffea/discussions/705
        pt_gen_axis = hist.axis.Variable(ptbins, name="pt_gen", overflow=True, underflow=True, label=r"$p_{T,gen}$")
        ptresponse_axis = hist.axis.Regular( 100, 0, 2.5, overflow=True, underflow=True, name="ptresponse", label="RECO / GEN response")
        jeteta_axis = hist.axis.Variable(etabins, name="jeteta", label=r"Jet $\eta$")

        # self.for_memory_testing()
        output = {'ptresponse_'+samp:hist.Hist(pt_gen_axis, ptresponse_axis, jeteta_axis, storage="weight", name="Counts")
                  for samp in flavors}
        # self.for_memory_testing()
        # To calculate the mean recopt, store only the sums of values for each bin.
        # Thus it takes much less space than storing the whole reco_pt distribution.  
        for samp in flavors:
            output['reco_pt_sumwx_'+samp] = hist.Hist(pt_gen_axis, jeteta_axis, storage="weight", name="Counts")
#         self.for_memory_testing()
        
        cutflow_axis = hist.axis.StrCategory([], growth=True, name="cutflow", label="Cutflow Scenarios")
        output['cutflow_events'] = hist.Hist(cutflow_axis, storage="weight", label="N events")
        output['cutflow_jets'] = hist.Hist(cutflow_axis, storage="weight", label="N jets")
        output['sum_weights'] = hist.Hist(cutflow_axis, storage="weight", label="sum of weights")

        dataset = events.metadata['dataset']
    
        ############ Event Cuts ############
        # apply npv cuts
        output['cutflow_events'].fill(cutflow='all_events', weight=len(events))
        
        npvCut = (events.PV.npvsGood > 0)
        pvzCut = (np.abs(events.PV.z) < 24)
        rxyCut = (np.sqrt(events.PV.x*events.PV.x + events.PV.y*events.PV.y) < 2)
        # dzvtxCut = (np.abs(events.GenVtx.z-events.PV.z)<0.2)
        
        selectedEvents = events[npvCut & pvzCut & rxyCut ] #& dzvtxCut
        output['cutflow_events'].fill(cutflow='gen vertex cut', weight=len(selectedEvents))
        # get GenJets and Jets
        jets = selectedEvents.Jet
        output['cutflow_jets'].fill(cutflow='all_jets', weight=ak.sum(ak.num(jets)))

        ########### Redo the flavour tagging if neccesarry. LHE Flavour2 derivation has to be done before the jet cuts  ###########
        #### Some samples have a missing LHE flavour infomration ####
        if (not 'LHEPart' in selectedEvents.fields) and ('LHE_flavour' in self.jetflavour):
            raise ValueError(f"jet flavour is chosen as {self.jetflavour}, but the sample does not contain 'LHEPart' "+
                                 ", so the jet flavour cannot be recalculated.")
             
        if self.jetflavour=='LHE_flavour_2':
                jets = get_LHE_flavour_2(jets, selectedEvents)

        ############ Jet selection ###########
        # Require that at least one gen jet is matched
        jet_gen_match_mask = ~ak.is_none(jets.matched_gen,axis=1)
        selected_jets = jets[jet_gen_match_mask]
        output['cutflow_jets'].fill(cutflow='gen matched', weight=ak.sum(ak.num(selected_jets)))

        ############ Apply Jet energy corrections on the jets ###########
        # define variables needed for corrected jets
        # https://coffeateam.github.io/coffea/notebooks/applying_corrections.html#Applying-energy-scale-transformations-to-Jets
        ## raw - subtracting back the corrections applying when generating the NanoAOD
        selected_jets['pt_raw'] = (1 - selected_jets['rawFactor']) * selected_jets['pt']     #raw pt. pt before the corrections are applied to data
        selected_jets['mass_raw'] = (1 - selected_jets['rawFactor']) * selected_jets['mass']
        selected_jets['pt_gen'] = ak.values_astype(ak.fill_none(selected_jets.matched_gen.pt, 0), np.float32)
        selected_jets['rho'] = ak.broadcast_arrays(selectedEvents.fixedGridRhoFastjetAll, selected_jets.pt)[0]
        events_cache = selectedEvents.caches[0]

        reco_jets = self.jet_factory.build(selected_jets, lazy_cache=events_cache)

        leptons, tightelectrons, tightmuons = select_leptons(selectedEvents)
        # ###### Event selection based on leptons: 2 (0/1/2) reco leptons for DY (TTBAR semilep) ######
        # # cuts on DY and ttbar based on L3Res selections https://twiki.cern.ch/twiki/bin/view/CMS/L3ResZJet
        if self.cfg["good_lepton_cut"]["apply"]==True:
            selectedEvents, reco_jets, leptons, tightelectrons, tightmuons = good_lepton_cut(reco_jets, selectedEvents, dataset, leptons, tightelectrons, tightmuons)
        output['cutflow_events'].fill(cutflow='lepton selection', weight=len(selectedEvents))

        # Require tight lepton veto id on jets = no matched (dressed) leptons in the jet;
        # Leptons are also reconstructed as jets with just one (or more) particle, so it is important to remove them
        if self.cfg["tight_lepton_veto_id"]["apply"]==True:
            reco_jets[(reco_jets.jetId >> 2 & 1)==1] ### tight lepton veto id
        output['cutflow_jets'].fill(cutflow='tight lep. id', weight=ak.sum(ak.num(reco_jets)))

        if self.cfg["recolep_drcut"]["apply"]==True:
            reco_jets = recolep_drcut(reco_jets, leptons, leptons)
        output['cutflow_jets'].fill(cutflow=r'$\Delta R$'+' cut with leptons', weight=ak.sum(ak.num(reco_jets)))

        cut_tmp = self.cfg["jet_pt_cut"]
        if cut_tmp["apply"]==True:
            reco_jets = jet_pt_cut(reco_jets, cut_tmp["mingenjetpt"])
        output['cutflow_jets'].fill(cutflow='jetpt cut', weight=ak.sum(ak.num(reco_jets)))

        # redo the jet matching with potentially lower dr cut than matched automatically
        cut_tmp = self.cfg["reco_jetMCmatching"]
        if cut_tmp["apply"]==True:
            reco_jets = jetMCmatching(reco_jets, **cut_tmp)
        output['cutflow_jets'].fill(cutflow='matched gen cut', weight=ak.sum(ak.num(reco_jets)))
        ######### Alpha cut = cut on the additional jet activity  ############    
        # Not used since run 2 because the large pileup causes a bias    
        cut_tmp = self.cfg["leading_jet_and_alpha_cut"]
        if cut_tmp["apply"]==True:
            reco_jets, selectedEvents = leading_jet_and_alpha_cut(reco_jets, leptons, selectedEvents, dataset, **remove_apply(cut_tmp))

        cut_tmp = self.cfg["select_Nth_jet"]
        if cut_tmp["apply"]==True:
            reco_jets, selectedEvents = select_Nth_jet(reco_jets, selectedEvents, cut_tmp["N"])
        output['cutflow_jets'].fill(cutflow=r'$\alpha$ cut'+'\nleading jets', weight=ak.sum(ak.num(reco_jets)))
        output['cutflow_events'].fill(cutflow=r'$\alpha$ cut',       weight=len(selectedEvents))

        # # Cut on overlapping jets
        cut_tmp = self.cfg["jet_iso_cut"]
        if cut_tmp["apply"]==True:
            reco_jets = jet_iso_cut(reco_jets, **remove_apply(cut_tmp))
        output['cutflow_jets'].fill(cutflow='iso jets', weight=ak.sum(ak.num(reco_jets)))
        gen_jets = reco_jets.matched_gen

        ############ Derive LHE flavour   ###########
        if self.jetflavour=='LHE_flavour':
            reco_jets = get_LHE_flavour(reco_jets, selectedEvents)      
        
        jet_flavour = reco_jets[self.jetflavour] 

        ########### Split the samples into jet flavours ###############
        shapes_jets = ak.num(gen_jets.pt) #for event weights
        gen_jetpt  = ak.flatten(gen_jets.pt).to_numpy( allow_missing=True)
        gen_jeteta = ak.flatten(gen_jets.eta).to_numpy( allow_missing=True)
        jetpt      = ak.flatten(reco_jets.pt).to_numpy( allow_missing=True)
        # jeteta     = ak.flatten(reco_jets.eta).to_numpy( allow_missing=True)

        # ptresponse_np = jetpt / gen_jetpt 
        # correction_pos_pt = (len(self.ptbins_closure)
        #                       - np.count_nonzero(np.array(gen_jetpt, ndmin=2).transpose() < self.ptbins_closure, axis=1))
        # correction_pos_eta = (len(self.etabins_closure)
        #                       - np.count_nonzero(np.abs(np.array(gen_jeteta, ndmin=2).transpose()) < self.etabins_closure, axis=1))
        
        ptresponse_np = jetpt / gen_jetpt #/ self.closure_corr[correction_pos_pt, correction_pos_eta]
        
        if 'LHEWeight' not in selectedEvents.fields: ### no LHEWeight.originalXWGTUP stored in standalone Pythia8 but Generator.weight instead
            gen_weights = selectedEvents.Generator.weight
        else:
            gen_weights = selectedEvents.LHEWeight.originalXWGTUP
        
        if self.cfg["use_gen_weights"]:
            weights = gen_weights
        else:
            weights = np.ones(len(selectedEvents))

        if self.cfg["use_pu_weights"]:
            weights = weights*self.pucorr['Collisions18_UltraLegacy_goldenJSON'].evaluate(selectedEvents.Pileup.nTrueInt, "nominal")
        weights_jet = np.repeat(weights, shapes_jets)


        masks = {flav: ak.flatten((jet_flavour == self.flavor2partonNr[flav] )).to_numpy( allow_missing=True)
                 for flav in flavors if 'unmatched' not in flav}
        from functools import reduce
        ## find the jets that are not taggeed as any of the flavours
        masks['unmatched'] = reduce(lambda x, y: x+y, masks.values()) == 0
        # for flav in flavors:
        #      print(f"sum masks for flav {flav} = {np.sum(masks[flav])}.")
        #      print(f"sum masks2 for flav {flav} = {np.sum(masks2[flav])}.")
        # print(f"sum masks all { np.sum([ masks[flav] for flav in flavors])}.")
        # print(f"len masks =  { len( masks['b'])}.")
        # print(f"len iso jets  =  { ak.sum(ak.num(selected_jets))}.")
        # print(f"sum masks for flav {'b'} = {np.sum(masks['b'])}.")
        # print(f"len reco jets  =  { ak.sum(ak.num(reco_jets))}.")
        # print(f"len gen jets  =  { ak.sum(ak.num(gen_jets))}.")

        ptresponses     = { flav: ptresponse_np[masks[flav]]        for flav in flavors }
        gen_jetpts      = { flav: gen_jetpt[masks[flav]]            for flav in flavors }
        gen_jetetas     = { flav: gen_jeteta[masks[flav]]           for flav in flavors }
        jetpts          = { flav: jetpt[masks[flav]]                for flav in flavors }
        # if self.cfg["use_weights"]==True:
        weights_jet     = { flav: weights_jet[masks[flav]]             for flav in flavors }
        # else:
        #     weights_jet     = { flav: np.ones_like(ptresponses[flav])   for flav in flavors }

        # print(f"len ptresponses {'b'} = {len(ptresponses['b'])}.")

        # print("Try to np:")
        # ak.flatten(gen_jetpt).to_numpy()
        # print("Try to np with Allow missing:")
        # ak.flatten(gen_jetpt).to_numpy(allow_missing=True)
        # print("Before filling:")
        # print("weights_jet = ", weights_jet)

        ########### Filling of the histograms ###############
        for flav in flavors:
            output['ptresponse_'+flav].fill(pt_gen=gen_jetpts[flav],
                                              jeteta=gen_jetetas[flav],
                                              ptresponse=ptresponses[flav],
                                              weight=weights_jet[flav]
                                             )
            
            output['reco_pt_sumwx_'+flav].fill(pt_gen=gen_jetpts[flav],
                                                 jeteta=gen_jetetas[flav],
                                                 weight=jetpts[flav]*weights_jet[flav]
                                                )
        # self.for_memory_testing()
        output['sum_weights'].fill(cutflow='sum_weights', weight=ak.sum(gen_weights))
        # allflav = [key for key in output.keys() if 'ptresponse' in key]
        # print(f"len iso jets  =  { sum([output[key].sum().value for key in allflav]) }.")
        # print(f"sum weights  =  { sum([output[key].sum().value for key in allflav]) }.")
        # print(f"output sum for {'b'} = {output['ptresponse_b'].sum().value}.")

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator