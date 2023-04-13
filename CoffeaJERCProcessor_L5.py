#CoffeaJERCProcessor_L5.py

from memory_profiler import profile
# from guppy import hpy
# h = hpy()
# print(h.heap())

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

from count_2d import count_2d
import hist
import awkward as ak
# from coffea import some_test_func
# some_test_func.test_func()


manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]

# ptbins = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 27, 30, 35, 40, 45, 57, 72, 90, 120, 
        # 150, 200, 300, 400, 550, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 10000 ])
ptbins = np.array([15, 17, 20, 23, 27, 30, 35, 40, 45, 57, 72, 90, 120, 
        150, 200, 300, 400, 550, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 10000 ])

etabins = np.array([-5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489,
                        -3.314, -3.139, -2.964, -2.853,  -2.65,   -2.5, -2.322, -2.172, -2.043,  -1.93, 
                        -1.83,  -1.74, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044,
                        -0.957, -0.879, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174,
                        -0.087,  0,  0.087,  0.174,  0.261,  0.348,  0.435,  0.522,  0.609,  0.696, 
                        0.783,  0.879,  0.957,  1.044,  1.131,  1.218,  1.305,  1.392,  1.479,  1.566, 
                        1.653,   1.74,   1.83,   1.93,  2.043,  2.172,  2.322,    2.5,   2.65,  2.853,
                        2.964,  3.139,  3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,
                        4.716,  4.889, 5.191])


# etabins = np.array([-5, -3, -2.5, -1.3, 0, 1.3, 2.5, 3, 5])


"""@TTbarResAnaHadronic Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. 
"""
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
    print(f"awkward version {ak.__version__}")
            
    @property
    def accumulator(self):
        return self._accumulator

    @profile    
    def for_memory_testing(self):
        a=1
        
#     @profile    
    def process(self, events):

        flavors = self.flavors        
        # flavour_axis = hist.axis.StrCategory(flavors, growth=False, name="jet_flav", label=r"jet_flavour")  ###not completelly sure if defining an axis is better than doing through a dictionary of flavors. See, https://github.com/CoffeaTeam/coffea/discussions/705
        pt_gen_axis = hist.axis.Variable(ptbins, name="pt_gen", overflow=True, underflow=True, label=r"$p_{T,gen}$")
        ptresponse_axis = hist.axis.Regular( 100, 0, 2.5, overflow=True, underflow=True, name="ptresponse", label="RECO / GEN response")
        jeteta_axis = hist.axis.Variable(etabins, name="jeteta", label=r"Jet $\eta$")

        self.for_memory_testing()
        output = {'ptresponse_'+samp:hist.Hist(pt_gen_axis, ptresponse_axis, jeteta_axis, storage="weight", name="Counts")
                  for samp in flavors}
        # self.for_memory_testing()
        ### Store only the sums of values. Much simpler than storing the whole reco_pt histogram
        for samp in flavors:
            output['reco_pt_sumwx_'+samp] = hist.Hist(pt_gen_axis, jeteta_axis, storage="weight", name="Counts")
#         self.for_memory_testing()
        
        cutflow_axis = hist.axis.StrCategory([], growth=True, name="cutflow", label="Cutflow Scenarios")
        output['cutflow'] = hist.Hist(cutflow_axis, storage="weight", label="Counts")

        dataset = events.metadata['dataset']
    
        # Event Cuts
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
        
        lhe_part_exists = False
#         try:
#                 events.LHEPart
#                 lhe_part_exists = True
#         except(AttributeError):
#                 print("LHEPart doesn't exist")
#         except:
#                 print("Error at LHE Part")

        ########### LHE Flavour2 derivation ###########
        ''' Algorithm of LHE_Flavour2:
        Cuts all the outgoing LHE particles that have pdgId as quarks (except top) and gluons.
        For each LHE particle finds the closest jet and gives the jet its flavour.
        If a jet is marked by two or more LHE particles: assign -999
        '''
#         lhe_part_exists = True 
        if lhe_part_exists:
                LHE_flavour_2 = ak.zeros_like(jets.hadronFlavour)
                jet_shape2 = ak.num(jets.hadronFlavour)

                ## have to work with flattened objects as awkwards doesn not allow to modify it's entries
                LHE_flavour_np_2 = ak.flatten(LHE_flavour_2).to_numpy().copy()

                LHEPart = selectedEvents.LHEPart
                absLHEid = np.abs(LHEPart.pdgId)
                LHE_outgoing = LHEPart[(LHEPart.status==1) & ((absLHEid < 6) | (absLHEid == 21))]

                drs, [LHE_match, jets_match] = LHE_outgoing.metric_table(jets, return_combinations=True, axis=1)

                arms = ak.argmin(drs, axis=2) ## for each event, for each LHE particle, the closest jet index
                cums = np.cumsum(jet_shape2)[:-1]
                cums = np.append(0,cums)
                arms_flat = arms + cums ### positions of the matchet jets in the flattened list
                arms_np = ak.flatten(arms_flat).to_numpy().data
                LHE_match_flat = ak.flatten(LHE_match[:,:,:1].pdgId,axis=1)
                
                aa = count_2d(arms, ak.ArrayBuilder())
                aa_np = ak.flatten(aa).to_numpy()

                LHE_flavour_np_2 = ak.flatten(LHE_flavour_2).to_numpy().copy()
                LHE_flavour_np_2[arms_np[ak.num(LHE_match_flat)>0][aa_np==1]] = ak.flatten(LHE_match_flat)[aa_np==1]
                ### Some LHE particles might point to the same LHE partons. Those are kept unmatched.
                LHE_flavour_np_2[arms_np[ak.num(LHE_match_flat)>0][aa_np>1]] = -999 

                jets["LHE_Flavour2"] = ak.unflatten(LHE_flavour_np_2, jet_shape2)

        ############ Jet selection ###########
        # Cut if no matching gen jet found
        jet_gen_match_mask = ~ak.is_none(jets.matched_gen,axis=1)
        # At least one matched (dressed) electron/muon found;
        # each jet has two slots for matched electrons available. Check that both are None. 

        genpart = selectedEvents.GenPart
        lepton_mask = (
                ((np.abs(genpart.pdgId) == 11) | (np.abs(genpart.pdgId) == 13) | (np.abs(genpart.pdgId) == 15 ))
                & (genpart.statusFlags>>13&1 == 1) 
                & (genpart.statusFlags&1 == 1)
        )
        genpart = genpart[lepton_mask]
        drs = jets.metric_table(genpart, return_combinations=False, axis=1 )
        matched_with_promt_lep = np.any((drs<0.4),axis=2)
        jet_mask = jet_gen_match_mask  & np.logical_not(matched_with_promt_lep)
            
        selected_jets = jets[jet_mask]
        output['cutflow'].fill(cutflow='gen_matched+no_dressed_lep', weight=ak.sum(ak.num(selected_jets)))

        ## select only the first three jets on QCD samples to avoid effects due to a non-physical jet spectrum 
        selected_jets = selected_jets #[:,0:3]
        jet_pt_mask = selected_jets.matched_gen.pt>15
        ## funny workaround to change the ak.type of jet_pt_mask from '10 * var * ?bool' to '10 * var * bool'
        ## otherwise after the correction .matched_gen field is not found.
        jet_pt_mask_shape = ak.num(jet_pt_mask)
        jet_pt_mask_np = ak.flatten(jet_pt_mask).to_numpy()
        jet_pt_mask = ak.unflatten(jet_pt_mask_np.data, jet_pt_mask_shape)
        sel_jets = selected_jets[jet_pt_mask]
        
                               
        output['cutflow'].fill(cutflow='jetpt>15', weight=ak.sum(ak.num(sel_jets)))
        
        
                # Cut on overlapping jets
        drs, _ = sel_jets.metric_table(sel_jets, return_combinations=True, axis=1)
        jet_iso_mask = ~ ak.any((1e-10<drs) & (drs<0.8), axis=2 )
        sel_jets = sel_jets[jet_iso_mask]
        
        output['cutflow'].fill(cutflow='iso jets', weight=ak.sum(ak.num(sel_jets)))

        ############ Apply Jet energy corrections on the jets ###########
        # define variables needed for corrected jets
        # https://coffeateam.github.io/coffea/notebooks/applying_corrections.html#Applying-energy-scale-transformations-to-Jets
        ## raw - subtracting back the corrections applying when generating the NanoAOD
        sel_jets['pt_raw'] = (1 - sel_jets['rawFactor']) * sel_jets['pt']     #raw pt. pt before the corrects applied to data
        sel_jets['mass_raw'] = (1 - sel_jets['rawFactor']) * sel_jets['mass']
        sel_jets['pt_gen'] = ak.values_astype(ak.fill_none(sel_jets.matched_gen.pt, 0), np.float32)
        sel_jets['rho'] = ak.broadcast_arrays(selectedEvents.fixedGridRhoFastjetAll, sel_jets.pt)[0]
        events_cache = selectedEvents.caches[0]

        reco_jets = self.jet_factory.build(sel_jets, lazy_cache=events_cache)
        gen_jets = reco_jets.matched_gen

        
        ############ Derive LHE flavour   ###########
        #### (does not work for the QCD data sample because it is lacking LHE flav  ) ####
        """ Algorithm for the flavour derivation:
        - Find all the matched outgoing LHE particles within dR<0.4
        - If there is at least one LHE particle with b flavour (bbar flavour), set LHE_flavour to 5 (-5). If both b and bbar are found, set LHE_flavour=0
        - If there is no b quark then: 
        If there is at least one LHE particle with c flavour (cbar flavour), set LHE_flavour to 4 (-4).
        If both are found, set LHE_flavour=0.
        - If none of the above:
        Assign the flavour of the hardest selected LHE particle.
        """

        lhe_part_exists = False
        if lhe_part_exists:
                LHE_flavour = ak.zeros_like(reco_jets.hadronFlavour)
                jet_shape = ak.num(reco_jets.hadronFlavour)
                LHE_flavour_np = ak.flatten(LHE_flavour).to_numpy()

                LHEPart = selectedEvents.LHEPart
                absLHEid = np.abs(LHEPart.pdgId)
                LHE_outgoing = LHEPart[(LHEPart.status==1) & ((absLHEid < 6) | (absLHEid == 21))]
                drs, [_, LHE_match] = reco_jets.metric_table(LHE_outgoing, return_combinations=True, axis=1)
                LHE_match = LHE_match[drs<0.4]

                b_criteria      = ak.any((LHE_match.pdgId==5),axis=2)
                bbar_criteria   = ak.any((LHE_match.pdgId==-5),axis=2)
                c_criteria      = ak.any((LHE_match.pdgId==4),axis=2)
                cbar_criteria   = ak.any((LHE_match.pdgId==-4),axis=2)
                rest_crit = ((LHE_match.pdgId==1) | (LHE_match.pdgId==2) | (LHE_match.pdgId==3) | (LHE_match.pdgId==-1)
        |                   (LHE_match.pdgId==-2) | (LHE_match.pdgId==-3) | (LHE_match.pdgId==21))
                rest_match_candidates = LHE_match[rest_crit]
                rest_match = rest_match_candidates[ak.argmax(rest_match_candidates.pt, axis=2, keepdims=True )]
                #for some reason it does not work with just one ak.flatten
                rest_flav_ids = ak.flatten(ak.flatten(rest_match.pdgId, axis=-1 )).to_numpy() 

                LHE_flavour_np[~rest_flav_ids.mask] = rest_flav_ids[~rest_flav_ids.mask]

                c_cri_np = ak.flatten(c_criteria & ~cbar_criteria).to_numpy()
                LHE_flavour_np[c_cri_np] = 4
                cbar_cri_np = ak.flatten(cbar_criteria & ~c_criteria).to_numpy()
                LHE_flavour_np[cbar_cri_np] = -4
                c_criteria_unknown = ak.flatten(cbar_criteria & c_criteria).to_numpy()
                LHE_flavour_np[c_criteria_unknown] = 0
                b_criteria_np = ak.flatten(b_criteria & ~bbar_criteria ).to_numpy()
                LHE_flavour_np[b_criteria_np] = 5
                bbar_criteria_np = ak.flatten(bbar_criteria & ~b_criteria).to_numpy()
                LHE_flavour_np[bbar_criteria_np] = -5
                b_criteria_unknown = ak.flatten(bbar_criteria & b_criteria).to_numpy()
                LHE_flavour_np[b_criteria_unknown] = 0

                reco_jets["LHE_Flavour"] = ak.unflatten(LHE_flavour_np, jet_shape)    
    
        # gen_jets = gen_jets
        # reco_jets = reco_jets         
        
        shapes_jets = ak.num(gen_jets.pt) #for event weights
        gen_jetpt  = ak.flatten(gen_jets.pt).to_numpy( allow_missing=True)
        gen_jeteta = ak.flatten(gen_jets.eta).to_numpy( allow_missing=True)
        jetpt      = ak.flatten(reco_jets.pt).to_numpy( allow_missing=True)
        jeteta     = ak.flatten(reco_jets.eta).to_numpy( allow_missing=True)
        
        # etabins_abs = etabins[(len(etabins)-1)//2:] ##the positive eta bins
        ptresponse_np = jetpt / gen_jetpt
        # correction_pos_pt = (len(self.ptbins_closure)
        #                       - np.count_nonzero(np.array(gen_jetpt, ndmin=2).transpose() < self.ptbins_closure, axis=1))
        # correction_pos_eta = (len(self.etabins_closure)
        #                       - np.count_nonzero(np.abs(np.array(gen_jeteta, ndmin=2).transpose()) < self.etabins_closure, axis=1))
        
        ptresponse_np = jetpt / gen_jetpt #/ self.closure_corr[correction_pos_pt, correction_pos_eta]
        
        jet_flavour = reco_jets.partonFlavour
#         jet_flavour = reco_jets.LHE_Flavour2
#         jet_flavour = reco_jets.LHE_Flavour
        
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
        # print(f"len iso jets  =  { ak.sum(ak.num(sel_jets))}.")
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
        
#         output['ptresponse'].fill(pt_gen=gen_jetpt,
#                                                    jeteta=gen_jeteta,
#                                                    ptresponse=ptresponse_np)
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
        self.for_memory_testing()

        # allflav = [key for key in output.keys() if 'ptresponse' in key]
        # print(f"len iso jets  =  { sum([output[key].sum().value for key in allflav]) }.")
        # print(f"sum weights  =  { sum([output[key].sum().value for key in allflav]) }.")
        # print(f"output sum for {'b'} = {output['ptresponse_b'].sum().value}.")

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator