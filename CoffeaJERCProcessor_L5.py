#CoffeaJERCProcessor.py

import scipy.stats as ss
from coffea import hist, processor, nanoevents
from coffea import util
import numpy as np
import pandas as pd
from coffea.jetmet_tools import JetCorrectionUncertainty # FactorizedJetCorrector
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor



import awkward as ak
#from coffea.nanoevents.methods import nanoaod
# from coffea.nanoevents.methods import candidate
# from coffea.nanoevents.methods import vector



manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]

ptbins = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 27, 30, 35, 40, 45, 57, 72, 90, 120, 
        150, 200, 300, 400, 550, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 10000 ])

etabins = np.array([-5, -3, -2.5, -1.3, 0, 1.3, 2.5, 3, 5])


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

        
        subsamples = ['b', 'c', 'ud', 's', 'g']
        self.subsamples = subsamples
        acc_dict = {'ptresponse_'+samp: hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, ptresponse_axis) for samp in subsamples}
        acc_dict['ptresponse']  = hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, ptresponse_axis)
        acc_dict['jetpt']       = hist.Hist("Counts", dataset_axis, jetpt_axis)
        acc_dict['jeteta']      = hist.Hist("Counts", dataset_axis, jeteta_axis)
        acc_dict['cutflow']     = processor.defaultdict_accumulator(int)

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
        
        print(evaluator)
        print(evaluator.keys())
        
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

    def process(self, events):
        
        output = self.accumulator.identity()
        
        dataset = events.metadata['dataset']
    
        # Event Cuts
        # apply npv cuts
        npvCut = (events.PV.npvsGood > 0)
        pvzCut = (np.abs(events.PV.z) < 24)
        rxyCut = (np.sqrt(events.PV.x*events.PV.x + events.PV.y*events.PV.y) < 2)
        
        selectedEvents = events[npvCut & pvzCut & rxyCut]

        # get GenJets and Jets
        jets = selectedEvents.Jet
#         gen_jets = jets.matched_gen #[:,0:2]
        
#         print("N jets before cuts tot = ", ak.sum(ak.num(jets)))
        ############ Jet selection ###########
        # Cut if no matching gen jet found
        jet_gen_match_mask = ~ak.is_none(jets.matched_gen,axis=1)
        # At least one matched (dresseds) electron/muon found
        dressed_electron_mask = ak.sum(ak.is_none(jets.matched_electrons,axis=2), axis=2)==2
        dressed_muon_mask     = ak.sum(ak.is_none(jets.matched_muons,axis=2), axis=2)==2
        # pt mask applied after the corrections
        jet_mask = jet_gen_match_mask & dressed_electron_mask & dressed_muon_mask  #& jet_pt_mask
        
        selected_jets = jets[jet_mask]
        
        # Cut on overlapping jets
        drs, _ = selected_jets.metric_table(selected_jets, return_combinations=True, axis=1)
        jet_iso_mask = ~ ak.any((1e-10<drs) & (drs<0.8), axis=2 )

        
#         print("N jets before iso cut = ", ak.sum(ak.num(jets)))
        sel_jets = selected_jets[jet_iso_mask]
        
#         print("jet_pt_mask/ jet_gen_match_mask/ dressed_electron_mask/ dressed_muon_mask/ jet_iso_mask/ jet_mask")
#         print(ak.sum(jet_pt_mask))
#         print(ak.sum(jet_gen_match_mask))
#         print(ak.sum(dressed_electron_mask))
#         print(ak.sum(dressed_muon_mask))
#         print(ak.sum(jet_iso_mask))
#         print(ak.sum(jet_mask)    )
#         print("N jets tot = ", ak.sum(ak.num(jets)))
#         print("By ev = ", ak.num(jets)[:20])

        ############ Apply Jet energy corrections on the jets ###########
        # define variables needed for corrected jets
        # https://coffeateam.github.io/coffea/notebooks/applying_corrections.html#Applying-energy-scale-transformations-to-Jets
        ## raw - subtracting back the corrections applying when generating the NanoAOD
        sel_jets['pt_raw'] = (1 - sel_jets['rawFactor']) * sel_jets['pt']     #raw pt. pt before the corrects applied to data
        sel_jets['mass_raw'] = (1 - sel_jets['rawFactor']) * sel_jets['mass']
        sel_jets['pt_gen'] = ak.values_astype(ak.fill_none(sel_jets.matched_gen.pt, 0), np.float32)
        sel_jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, sel_jets.pt)[0]
        events_cache = events.caches[0]

        reco_jets = self.jet_factory.build(sel_jets, lazy_cache=events_cache)
        reco_jets = reco_jets[reco_jets.pt>20]
        gen_jets = reco_jets.matched_gen

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
        jetpt             = ak.flatten(reco_jets.pt).to_numpy( allow_missing=True)
        jeteta            = ak.flatten(reco_jets.eta).to_numpy( allow_missing=True)
        
        etabins_abs = etabins[(len(etabins)-1)//2:] ##the positive eta bins
        ptresponse_np = jetpt / gen_jetpt
        correction_pos_pt = len(ptbins) - np.count_nonzero(np.array(gen_jetpt, ndmin=2).transpose() < ptbins, axis=1)
        correction_pos_eta = len(etabins_abs) - np.count_nonzero(np.abs(np.array(gen_jeteta, ndmin=2).transpose()) < etabins_abs, axis=1)
        
#         print("gen_jetpt = ", gen_jetpt)
#         print("gen_jetpt np = ", gen_jetpt)
        
#         ptresponse_np = jetpt / gen_jetpt
        ptresponse_np = jetpt / gen_jetpt / self.closure_corr[correction_pos_pt, correction_pos_eta]
#         ptresponse = jetpt / gen_jetpt 
        
#         print("corr pos = ", [correction_pos_pt[:10], correction_pos_eta[:10]])
#         print("closure = ", self.closure_corr[correction_pos_pt[:10], correction_pos_eta[:10]])
#         print("ptresp = ", ptresponse)
#         print("ptresp_np = ", ptresponse_np)
        
        
        subsamples = self.subsamples
        masks = {}
        masks['b'] = ak.flatten((reco_jets.partonFlavour==5) | (reco_jets.partonFlavour==-5)).to_numpy( allow_missing=True)
        masks['c'] = ak.flatten((reco_jets.partonFlavour==4) | (reco_jets.partonFlavour==-4)).to_numpy( allow_missing=True)
        masks['ud'] = ak.flatten((reco_jets.partonFlavour==1) | (reco_jets.partonFlavour==-1) |
                     (reco_jets.partonFlavour==2) | (reco_jets.partonFlavour==-2) ).to_numpy( allow_missing=True)
        masks['s'] = ak.flatten((reco_jets.partonFlavour==3) | (reco_jets.partonFlavour==-3) ).to_numpy( allow_missing=True)
        masks['g'] = ak.flatten(reco_jets.partonFlavour==21).to_numpy( allow_missing=True)

        ptresponses     = { sample: ptresponse_np[masks[sample]]             for sample in subsamples }
        gen_jetpts      = { sample: gen_jetpt[masks[sample]]       for sample in subsamples }
        gen_jetetas     = { sample: gen_jeteta[masks[sample]]      for sample in subsamples }

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



