#CoffeaJERCProcessor_L5.py

from memory_profiler import profile
# from guppy import hpy
# h = hpy()
# print(h.heap())

import sys
sys.path.insert(0,'/afs/cern.ch/user/a/anpotreb/top/JERC/JMECoffea')
sys.path

# from os import listdir
# listdir('.')
# listdir('./coffea')

# import scipy.stats as ss
from coffea import hist, processor, nanoevents
from coffea import util
import numpy as np
import pandas as pd
from coffea.jetmet_tools import JetCorrectionUncertainty # FactorizedJetCorrector
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor

from coffea.count_2d import count_2d
# from coffea import some_test_func
# some_test_func.test_func()



import awkward as ak
#from coffea.nanoevents.methods import nanoaod
# from coffea.nanoevents.methods import candidate
# from coffea.nanoevents.methods import vector



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
        
        
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        # cats_axis = hist.Cat("anacat", "Analysis Category")
       
        jetpt_axis = hist.Bin("pt", r"$p_T$", ptbins)
        ptresponse_axis = hist.Bin("ptresponse", "RECO / GEN response", 100, 0, 2.5)
        # jetmass_axis = hist.Bin("jetmass", r"Jet $m$ [GeV]", 50, 0, 500)
        jeteta_axis = hist.Bin("jeteta", r"Jet $\eta$", etabins)
        # jetphi_axis = hist.Bin("jetphi", r"Jet $\phi$", 50, -np.pi, np.pi)

        
        subsamples = ['b', 'c', 'u', 'd', 's', 'g', 'bbar', 'cbar', 'ubar', 'dbar', 'sbar']
        self.subsamples = subsamples
        acc_dict = {'ptresponse_'+samp: hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, ptresponse_axis) for samp in subsamples}
        acc_dict['ptresponse']  = hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, ptresponse_axis)
        acc_dict['jetpt']       = hist.Hist("Counts", dataset_axis, jetpt_axis)
        acc_dict['jeteta']      = hist.Hist("Counts", dataset_axis, jeteta_axis)
        
#         acc_dict['cutflow']     = processor.defaultdict_accumulator(dict)
        acc_dict['cutflow']    = processor.defaultdict_accumulator(int)

        self._accumulator = processor.dict_accumulator(acc_dict)
        
        
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
        
        print("evaluator = ", evaluator)
        print("evaluator keys = ", evaluator.keys())
        
        jec_inputs = {name: evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)
        
        
        ### more possibilities are available if you send in more pieces of the JEC stack
        # mc2016_ak8_jxform = JECStack(["more", "names", "of", "JEC parts"])
        
#         self.corrector = FactorizedJetCorrector(
#             Summer20UL18_V2_MC_L1FastJet_AK4PFchs=evaluator['Summer20UL18_V2_MC_L1FastJet_AK4PFchs'],
#             Summer20UL18_V2_MC_L2Relative_AK4PFchs=evaluator['Summer20UL18_V2_MC_L2Relative_AK4PFchs'],
#             Summer20UL18_V2_MC_L3Absolute_AK4PFchs=evaluator['Summer20UL18_V2_MC_L3Absolute_AK4PFchs'],
#         )

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

        df_csv = pd.read_csv('out_txt/Closure_L5_QCD.csv').set_index('etaBins')
        self.closure_corr = df_csv.to_numpy().transpose()
        self.closure_corr = np.pad(self.closure_corr,1,constant_values=1)
        
#         ### Uncomment to make a correction for closure
#         self.closure_corr = np.ones(self.closure_corr.shape)
        
#         print(dir(evaluator))
#         print()

            
    @property
    def accumulator(self):
        return self._accumulator

    @profile
    def process(self, events):
        
        output = self.accumulator.identity()

        dataset = events.metadata['dataset']
    
        # Event Cuts
        # apply npv cuts
        output['cutflow'][dataset+': all events'] = len(events)
        
        npvCut = (events.PV.npvsGood > 0)
        pvzCut = (np.abs(events.PV.z) < 24)
        rxyCut = (np.sqrt(events.PV.x*events.PV.x + events.PV.y*events.PV.y) < 2)
        
        selectedEvents = events[npvCut & pvzCut & rxyCut]

        # get GenJets and Jets
        jets = selectedEvents.Jet
        output['cutflow'][dataset+': all jets'] = ak.sum(ak.num(jets))
        
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
        # At least one matched (dressed) electron/muon found
        dressed_electron_mask = ak.sum(ak.is_none(jets.matched_electrons,axis=2), axis=2)==2
        dressed_muon_mask     = ak.sum(ak.is_none(jets.matched_muons,axis=2), axis=2)==2
        jet_mask = jet_gen_match_mask  & dressed_electron_mask & dressed_muon_mask
            
        selected_jets = jets[jet_mask]
        output['cutflow'][dataset+': gen matched + no dressed lep'] = ak.sum(ak.num(selected_jets))


        jet_pt_mask = selected_jets.matched_gen.pt>20
        ## funny workaround to change the ak.type of jet_pt_mask from '10 * var * ?bool' to '10 * var * bool'
        ## otherwise after the correction .matched_gen field is not found.
        jet_pt_mask_shape = ak.num(jet_pt_mask)
        jet_pt_mask_np = ak.flatten(jet_pt_mask).to_numpy()
        jet_pt_mask = ak.unflatten(jet_pt_mask_np.data, jet_pt_mask_shape)
        sel_jets = selected_jets[jet_pt_mask]
        
        output['cutflow'][dataset+': + jetpt>20'] = ak.sum(ak.num(sel_jets))
        
        
                # Cut on overlapping jets
        drs, _ = sel_jets.metric_table(sel_jets, return_combinations=True, axis=1)
        jet_iso_mask = ~ ak.any((1e-10<drs) & (drs<0.8), axis=2 )
        sel_jets = sel_jets[jet_iso_mask]
        
        output['cutflow'][dataset+': iso jets'] = ak.sum(ak.num(sel_jets))

        # print("jet_gen_match_mask/ dressed_electron_mask/ dressed_muon_mask/ jet_mask")
        
        # print(ak.sum(jet_gen_match_mask))
        # print(ak.sum(dressed_electron_mask))
        # print(ak.sum(dressed_muon_mask))
        # print(ak.sum(jet_mask)    )
        # print("N jets before cuts tot = ", ak.sum(ak.num(jets)))
        # print("By ev = ", ak.num(jets)[:20])
        # print("N jets after first cuts = ", ak.sum(ak.num(selected_jets)))
        # print("By ev = ", ak.num(jets)[:20])
        # print("jet_pt_mask = ", ak.sum(jet_pt_mask))
        # print("jet_iso_mask = ", ak.sum(jet_iso_mask))

        # print("N jets after sel = ", ak.sum(ak.num(sel_jets)))
        # print("By ev after sel = ", ak.num(sel_jets))

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

        lhe_part_exists = True
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


                # jetpt = jets.pt
        # corrected_jetpt = corrected_jets.pt

#         # Recalculate the MC jet matching
#         matchedJets = ak.cartesian([gen_jets, corrected_jets])
#         deltaR = matchedJets.slot0.delta_r(matchedJets.slot1)
#         matchedJets = matchedJets[deltaR < 0.2]
#         matchedJets = matchedJets[ak.num(matchedJets) > 0]

#         gen_jets = matchedJets.slot0
#         reco_jets = matchedJets.slot1        
    
        gen_jets = gen_jets
        reco_jets = reco_jets         
        
        gen_jetpt  = ak.flatten(gen_jets.pt).to_numpy( allow_missing=True)
        gen_jeteta = ak.flatten(gen_jets.eta).to_numpy( allow_missing=True)
        jetpt      = ak.flatten(reco_jets.pt).to_numpy( allow_missing=True)
        jeteta     = ak.flatten(reco_jets.eta).to_numpy( allow_missing=True)
        
        etabins_abs = etabins[(len(etabins)-1)//2:] ##the positive eta bins
        ptresponse_np = jetpt / gen_jetpt
        correction_pos_pt = len(ptbins) - np.count_nonzero(np.array(gen_jetpt, ndmin=2).transpose() < ptbins, axis=1)
        correction_pos_eta = len(etabins_abs) - np.count_nonzero(np.abs(np.array(gen_jeteta, ndmin=2).transpose()) < etabins_abs, axis=1)
        
        ptresponse_np = jetpt / gen_jetpt # / self.closure_corr[correction_pos_pt, correction_pos_eta]
        
#         jet_flavour = reco_jets.partonFlavour
#         jet_flavour = reco_jets.LHE_Flavour2
        jet_flavour = reco_jets.LHE_Flavour
        
        subsamples = self.subsamples
        masks = {}
        masks['b'] = ak.flatten((jet_flavour==5)).to_numpy( allow_missing=True)
        masks['c'] = ak.flatten((jet_flavour==4)).to_numpy( allow_missing=True)
        masks['d'] = ak.flatten(jet_flavour==2).to_numpy( allow_missing=True)
        masks['u'] = ak.flatten(jet_flavour==1).to_numpy( allow_missing=True)
        masks['s'] = ak.flatten((jet_flavour==3) ).to_numpy( allow_missing=True)
        masks['bbar'] = ak.flatten((jet_flavour==-5)).to_numpy( allow_missing=True)
        masks['cbar'] = ak.flatten((jet_flavour==-4)).to_numpy( allow_missing=True)
        masks['ubar'] = ak.flatten(jet_flavour==-1).to_numpy( allow_missing=True)
        masks['dbar'] = ak.flatten(jet_flavour==-2).to_numpy( allow_missing=True)
        masks['sbar'] = ak.flatten((jet_flavour==-3) ).to_numpy( allow_missing=True)
        masks['g'] = ak.flatten(jet_flavour==21).to_numpy( allow_missing=True)

        ptresponses     = { sample: ptresponse_np[masks[sample]]        for sample in subsamples }
        gen_jetpts      = { sample: gen_jetpt[masks[sample]]            for sample in subsamples }
        gen_jetetas     = { sample: gen_jeteta[masks[sample]]           for sample in subsamples }

        # print("Try to np:")
        # ak.flatten(gen_jetpt).to_numpy()
        # print("Try to np with Allow missing:")
        # ak.flatten(gen_jetpt).to_numpy(allow_missing=True)
        # print("Before filling:")

        ########### Filling of the histograms ###############

        output['jetpt'].fill(dataset = dataset, 
                        pt = gen_jetpt)

        output['jeteta'].fill(dataset = dataset, 
                        jeteta = jeteta)
        
        output['ptresponse'].fill(dataset=dataset, pt=gen_jetpt,
                                                   jeteta=gen_jeteta,
                                                   ptresponse=ptresponse_np)
  
        for sample in subsamples:
               output['ptresponse_'+sample].fill(dataset=dataset, pt=gen_jetpts[sample],
                                    jeteta=gen_jetetas[sample],
                                    ptresponse=ptresponses[sample])
            
        
        
        return output

    def postprocess(self, accumulator):
        return accumulator



