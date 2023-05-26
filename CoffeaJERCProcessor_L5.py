#CoffeaJERCProcessor_L5.py
''' coffea processor for calculating the jet enegy response and splitting the sample into jet flavors:
output: a dictionary over datasets of dictionaries over histograms.
Output histograms: ptresponse histogram, pt_reco histogram for each flavor and the cuflow
''' 

################## Switches #################
### Choose the jet flavour. Some samples have missing `partonFlavour`, so one has to redo the flavour matching oneself. Two different option were implemented:
### `LHE_flavour` starts from the jet and matches to the closest LHE particle.
### `LHE_flavour2` (a better option) starts from the LHE particle and matches to the jet 
jetflavour = 'partonFlavour'
# jetflavour = 'LHE_flavour'


from memory_profiler import profile
from common_binning import JERC_Constants

# workaround to get a locally installed coffea and awkwrd version using lch on lxplus
# comment out or replace the path if I happened to forget to remove these lines before pushing:
import sys
sys.path.insert(0,'/afs/cern.ch/user/a/anpotreb/top/JERC/coffea')
ak_path = '/afs/cern.ch/user/a/anpotreb/top/JERC/local-packages/'
if ak_path not in sys.path:
        sys.path.insert(0,ak_path)
# sys.path.insert(0,'/afs/cern.ch/user/a/anpotreb/top/JERC/JMECoffea')
# print("sys path = ", sys.path)
# from os import listdir
# listdir('.')
# listdir('./coffea')

from coffea import processor
import numpy as np
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor

from LHE_flavour import get_LHE_flavour, get_LHE_flavour_2
import hist
import awkward as ak

# from coffea import some_test_func
# some_test_func.test_func()


manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]
ptbins = np.array(JERC_Constants.ptBinsEdgesMCTruth())
etabins = np.array(JERC_Constants.etaBinsEdges_CaloTowers_full())

class Processor(processor.ProcessorABC):
    def __init__(self):        
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

    @profile    
    def for_memory_testing(self):
        a=1
        
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
        output['cutflow'] = hist.Hist(cutflow_axis, storage="weight", label="Counts")

        dataset = events.metadata['dataset']
    
        ############ Event Cuts ############
        # apply npv cuts
        output['cutflow'].fill(cutflow='all_events', weight=len(events))
        
        npvCut = (events.PV.npvsGood > 0)
        pvzCut = (np.abs(events.PV.z) < 24)
        rxyCut = (np.sqrt(events.PV.x*events.PV.x + events.PV.y*events.PV.y) < 2)
        
        selectedEvents = events[npvCut & pvzCut & rxyCut]
        output['cutflow'].fill(cutflow='selected_events', weight=len(selectedEvents))
        # get GenJets and Jets
        jets = selectedEvents.Jet
        output['cutflow'].fill(cutflow='all_jets', weight=ak.sum(ak.num(jets)))

        ########### Redo the flavour tagging if neccesarry. LHE Flavour2 derivation has to be done before the jet cuts  ###########
        #### Some samples have a missing LHE flavour infomration ####
        if (not 'LHEPart' in events.fields) and ('LHE_flavour' in jetflavour):
            raise ValueError(f"jet flavour is chosen as {jetflavour}, but the sample does not contain 'LHEPart' "+
                                 ", so the jet flavour cannot be recalculated.")
             
        if jetflavour=='LHE_flavour_2':
                jets = get_LHE_flavour_2(jets, selectedEvents)

        ############ Jet selection ###########
        # Require that at least one gen jet is matched
        jet_gen_match_mask = ~ak.is_none(jets.matched_gen,axis=1)
        selected_jets = jets[jet_gen_match_mask]
        output['cutflow'].fill(cutflow='gen_matched', weight=ak.sum(ak.num(selected_jets)))

        ############ Apply Jet energy corrections on the jets ###########
        # define variables needed for corrected jets
        # https://coffeateam.github.io/coffea/notebooks/applying_corrections.html#Applying-energy-scale-transformations-to-Jets
        ## raw - subtracting back the corrections applying when generating the NanoAOD
        selected_jets['pt_raw'] = (1 - selected_jets['rawFactor']) * selected_jets['pt']     #raw pt. pt before the corrects applied to data
        selected_jets['mass_raw'] = (1 - selected_jets['rawFactor']) * selected_jets['mass']
        selected_jets['pt_gen'] = ak.values_astype(ak.fill_none(selected_jets.matched_gen.pt, 0), np.float32)
        selected_jets['rho'] = ak.broadcast_arrays(selectedEvents.fixedGridRhoFastjetAll, selected_jets.pt)[0]
        events_cache = selectedEvents.caches[0]

        reco_jets = self.jet_factory.build(selected_jets, lazy_cache=events_cache)
        selectedEvents = selectedEvents #[:100]
        reco_jets = reco_jets #[:100]

        # print("---"*10)
        # print("Before alpha cut")
        # print("recojetpt = ", reco_jets.pt)
        # print("genjetpt = ", reco_jets.matched_gen.pt)

        # Require no matched (dressed) leptons in the jet;
        # Leptons are often misreconstructed as jets and can ruin the comparison between different samples.
        genpart = selectedEvents.GenPart
        lepton_mask = (
                ((np.abs(genpart.pdgId) == 11) | (np.abs(genpart.pdgId) == 13) | (np.abs(genpart.pdgId) == 15 ))
                & (genpart.statusFlags>>13&1 == 1) 
                & (genpart.statusFlags&1 == 1)
        )
        leptons = genpart[lepton_mask]

        drs = reco_jets.metric_table(leptons, return_combinations=False, axis=1 )
        matched_with_promt_lep = np.any((drs<0.4),axis=2)
        # jet_mask = np.logical_not(matched_with_promt_lep)
        # reco_jets = reco_jets[np.logical_not(matched_with_promt_lep)]
        # output['cutflow'].fill(cutflow='no_dressed_lep', weight=ak.sum(ak.num(reco_jets)))

        tight_jet = (reco_jets.jetId >> 2 & 1)
        # reco_jets = reco_jets[tight_jet==True]

        # At least one matched (dressed) electron/muon found
        # dressed_electron_mask = ak.sum(ak.is_none(reco_jets.matched_electrons,axis=2), axis=2)==2
        # dressed_muon_mask     = ak.sum(ak.is_none(reco_jets.matched_muons,axis=2), axis=2)==2
        # reco_jets = reco_jets[dressed_electron_mask & dressed_muon_mask]
        output['cutflow'].fill(cutflow='no_dressed_lep', weight=ak.sum(ak.num(reco_jets)))


        jet_pt_mask = reco_jets.matched_gen.pt>15
        ## funny workaround to change the ak.type of jet_pt_mask from '10 * var * ?bool' to '10 * var * bool'
        ## otherwise after the correction .matched_gen field is not found.
        jet_pt_mask_shape = ak.num(jet_pt_mask)
        jet_pt_mask_np = ak.flatten(jet_pt_mask).to_numpy()
        jet_pt_mask = ak.unflatten(jet_pt_mask_np.data, jet_pt_mask_shape)
        reco_jets = reco_jets[jet_pt_mask]
        output['cutflow'].fill(cutflow='jetpt>15', weight=ak.sum(ak.num(reco_jets)))

        ######### Alpha cut = cut on the additional jet activity  ############        
        # alphacut = 1.0
        if "QCD" in dataset:
            alphacut = 1.0 #if the alpha cut is different from the default
            # Correctly/safely treat the cases where there are less then 3 jets left after the cuts
            # select only the first three jets on QCD samples
            # to avoid effects due to a non-physical jet spectrum 
            reco_jetspt = ak.pad_none(reco_jets.pt, 3, axis=1, clip=True)
            # reco_jetspt = reco_jets.pt
            # print("---"*10)
            # print("Leading 3")
            # print("num recopt = ", ak.num(reco_jetspt))
            # print("recojetpt = ", reco_jetspt)
        # print("genjetpt = ", reco_jets.matched_gen.pt[:10])

            alpha = reco_jetspt[:,2]*2/(reco_jetspt[:,0]+reco_jetspt[:,1])
            alpha = ak.fill_none(alpha,0)

            reco_jets = reco_jets[alpha<alphacut][:,:3]
            selectedEvents = selectedEvents[alpha<alphacut]
        elif 'DY' in dataset:
            alphacut = 1.0 #if the alpha cut is different from the default
            reco_jetspt = ak.pad_none(reco_jets.pt, 2, axis=1, clip=True)
            # reco_jetspt = reco_jets.pt
            # print(reco_jetspt[:50])
            alpha = reco_jetspt[:,1]/ak.sum(leptons.pt,axis=1)
            alpha = ak.fill_none(alpha,0)
            reco_jets = reco_jets[alpha<alphacut][:,:2]
            selectedEvents = selectedEvents[alpha<alphacut]
        output['cutflow'].fill(cutflow=f'alpha cut; leading jets', weight=ak.sum(ak.num(reco_jets)))
        output['cutflow'].fill(cutflow=f'events, alpha cut',       weight=len(selectedEvents))
        
        # print("---"*10)
        # print("After alpha cut")
        # print("recojetpt = ", reco_jets.pt)
        # print("genjetpt = ", reco_jets.matched_gen.pt)
        # print("After pt_gen>15 cut")
        # print("recojetpt = ", reco_jets.pt)
        # print("genjetpt = ", reco_jets.matched_gen.pt   )

        # Cut on overlapping jets
        drs, _ = reco_jets.metric_table(reco_jets, return_combinations=True, axis=1)
        jet_iso_mask = ~ ak.any((1e-10<drs) & (drs<0.8), axis=2 )
        reco_jets = reco_jets[jet_iso_mask]
        output['cutflow'].fill(cutflow='iso jets', weight=ak.sum(ak.num(reco_jets)))
        gen_jets = reco_jets.matched_gen

        ############ Derive LHE flavour   ###########
        if jetflavour=='LHE_flavour':
            reco_jets = get_LHE_flavour(reco_jets, selectedEvents)      
        
        jet_flavour = reco_jets[jetflavour] 

        ########### Split the samples into jet flavours ###############
        shapes_jets = ak.num(gen_jets.pt) #for event weights
        gen_jetpt  = ak.flatten(gen_jets.pt).to_numpy( allow_missing=True)
        gen_jeteta = ak.flatten(gen_jets.eta).to_numpy( allow_missing=True)
        jetpt      = ak.flatten(reco_jets.pt).to_numpy( allow_missing=True)
        jeteta     = ak.flatten(reco_jets.eta).to_numpy( allow_missing=True)
        
        ptresponse_np = jetpt / gen_jetpt
        # correction_pos_pt = (len(self.ptbins_closure)
        #                       - np.count_nonzero(np.array(gen_jetpt, ndmin=2).transpose() < self.ptbins_closure, axis=1))
        # correction_pos_eta = (len(self.etabins_closure)
        #                       - np.count_nonzero(np.abs(np.array(gen_jeteta, ndmin=2).transpose()) < self.etabins_closure, axis=1))
        
        ptresponse_np = jetpt / gen_jetpt #/ self.closure_corr[correction_pos_pt, correction_pos_eta]
        
        try:
            weights = selectedEvents.LHEWeight.originalXWGTUP
        except AttributeError: ### no LHEWeight.originalXWGTUP in madgraph herwig but Generator.weight instead
            weights = selectedEvents.Generator.weight
    
        weights2 = np.repeat(weights, shapes_jets)


        masks = {flav: ak.flatten((jet_flavour == self.flavor2partonNr[flav] )).to_numpy( allow_missing=True)
                 for flav in flavors if 'unmatched' not in flav}
        from functools import reduce
        masks['unmatched'] = reduce(lambda x, y: x+y, masks.values()) == 0 ## find the jets that are not taggeed
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
        weights_jet     = { flav: weights2[masks[flav]]             for flav in flavors }

        # print(f"len ptresponses {'b'} = {len(ptresponses['b'])}.")

        # print("Try to np:")
        # ak.flatten(gen_jetpt).to_numpy()
        # print("Try to np with Allow missing:")
        # ak.flatten(gen_jetpt).to_numpy(allow_missing=True)
        # print("Before filling:")

        ########### Filling of the histograms ###############
        for flav in flavors:
            output['ptresponse_'+flav].fill(pt_gen=gen_jetpts[flav],
                                              jeteta=gen_jetetas[flav],
                                              ptresponse=ptresponses[flav],
#                                               weight=weights_jet[flav]
                                             )
            
            output['reco_pt_sumwx_'+flav].fill(pt_gen=gen_jetpts[flav],
                                                 jeteta=gen_jetetas[flav],
                                                 weight=jetpts[flav] #*weights_jet[flav]
                                                )
        # self.for_memory_testing()

        # allflav = [key for key in output.keys() if 'ptresponse' in key]
        # print(f"len iso jets  =  { sum([output[key].sum().value for key in allflav]) }.")
        # print(f"sum weights  =  { sum([output[key].sum().value for key in allflav]) }.")
        # print(f"output sum for {'b'} = {output['ptresponse_b'].sum().value}.")

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator