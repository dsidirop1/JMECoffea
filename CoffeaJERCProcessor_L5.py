#CoffeaJERCProcessor.py

import scipy.stats as ss
from coffea import hist, processor, nanoevents
from coffea import util
import numpy as np
import pandas as pd
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
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
        cats_axis = hist.Cat("anacat", "Analysis Category")
       
        jetpt_axis = hist.Bin("pt", r"$p_T$", ptbins)
        ptresponse_axis = hist.Bin("ptresponse", "RECO / GEN response", 100, 0, 5)
        jetmass_axis = hist.Bin("jetmass", r"Jet $m$ [GeV]", 50, 0, 500)
        jeteta_axis = hist.Bin("jeteta", r"Jet $\eta$", etabins)
        jetphi_axis = hist.Bin("jetphi", r"Jet $\phi$", 50, -np.pi, np.pi)

        subsamples = ['b', 'c', 'l', 'g']
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
            "* * Summer20UL18_V2_MC/Summer20UL18_V2_MC_L3Absolute_AK4PFchs.txt"
        ])
        ext.finalize()

        jec_stack_names = ["Summer20UL18_V2_MC_L1FastJet_AK4PFchs",
                           "Summer20UL18_V2_MC_L2Relative_AK4PFchs", 
                           "Summer20UL18_V2_MC_L3Absolute_AK4PFchs"]

        evaluator = ext.make_evaluator()
        
        print(evaluator)
        print(evaluator.keys())
        
        jec_inputs = {name: evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)
        
        
        ### more possibilities are available if you send in more pieces of the JEC stack
        # mc2016_ak8_jxform = JECStack(["more", "names", "of", "JEC parts"])
        
        self.corrector = FactorizedJetCorrector(
            Summer20UL18_V2_MC_L1FastJet_AK4PFchs=evaluator['Summer20UL18_V2_MC_L1FastJet_AK4PFchs'],
            Summer20UL18_V2_MC_L2Relative_AK4PFchs=evaluator['Summer20UL18_V2_MC_L2Relative_AK4PFchs'],
            Summer20UL18_V2_MC_L3Absolute_AK4PFchs=evaluator['Summer20UL18_V2_MC_L3Absolute_AK4PFchs'],
        )

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
        
        print(dir(evaluator))
        print()

            
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
        GenJets = selectedEvents.GenJet[:,0:2]
        jets = selectedEvents.Jet
        
        
        
        # define variables needed for corrected jets
        # https://coffeateam.github.io/coffea/notebooks/applying_corrections.html#Applying-energy-scale-transformations-to-Jets
        jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
        jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
        jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
        jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
        events_cache = events.caches[0]

        corrected_jets = self.jet_factory.build(jets, lazy_cache=events_cache)

        jetpt = jets.pt
        # corrected_jetpt = corrected_jets.pt

        # MC jet matching
        matchedJets = ak.cartesian([GenJets, corrected_jets])
        deltaR = matchedJets.slot0.delta_r(matchedJets.slot1)
        matchedJets = matchedJets[deltaR < 0.2]
        matchedJets = matchedJets[ak.num(matchedJets) > 0]

        
        
        
        jetpt = matchedJets.slot1.pt
        jeteta = matchedJets.slot1.eta


        
        matched_genjetpt = matchedJets.slot0.pt
        matched_genjeteta = matchedJets.slot0.eta
        
        
        matched_genjetpt_np  = ak.flatten(matched_genjetpt).to_numpy( allow_missing=True)
        matched_genjeteta_np = ak.flatten(matched_genjeteta).to_numpy( allow_missing=True)
        jetpt_np             = ak.flatten(jetpt).to_numpy( allow_missing=True)
        jeteta_np            = ak.flatten(jeteta).to_numpy( allow_missing=True)
        
        etabins_mod = etabins[(len(etabins)-1)//2:]
        ptresponse_np = jetpt_np / matched_genjetpt_np
        correction_pos_pt = len(ptbins) - np.count_nonzero(np.array(matched_genjetpt_np, ndmin=2).transpose() < ptbins, axis=1)
        correction_pos_eta = len(etabins_mod) - np.count_nonzero(np.abs(np.array(matched_genjeteta_np, ndmin=2).transpose()) < etabins_mod, axis=1)
        
#         print("matched_genjetpt = ", matched_genjetpt)
#         print("matched_genjetpt np = ", matched_genjetpt_np)
        
#         ptresponse_np = jetpt_np / matched_genjetpt_np
        ptresponse_np = jetpt_np / matched_genjetpt_np / self.closure_corr[correction_pos_pt, correction_pos_eta]
#         ptresponse = jetpt / matched_genjetpt 
        
#         print("corr pos = ", [correction_pos_pt[:10], correction_pos_eta[:10]])
#         print("closure = ", self.closure_corr[correction_pos_pt[:10], correction_pos_eta[:10]])
#         print("ptresp = ", ptresponse)
#         print("ptresp_np = ", ptresponse_np)
        
        
        subsamples = ['b', 'c', 'l', 'g']
        masks = {}
        masks['b'] = ak.flatten((matchedJets.slot0.partonFlavour==5) | (matchedJets.slot0.partonFlavour==-5)).to_numpy( allow_missing=True)
        masks['c'] = ak.flatten((matchedJets.slot0.partonFlavour==4) | (matchedJets.slot0.partonFlavour==-4)).to_numpy( allow_missing=True)
        masks['l'] = ak.flatten(((matchedJets.slot0.partonFlavour==1) | (matchedJets.slot0.partonFlavour==-1) |
                     (matchedJets.slot0.partonFlavour==2) | (matchedJets.slot0.partonFlavour==-2) |
                    (matchedJets.slot0.partonFlavour==3) | (matchedJets.slot0.partonFlavour==-3) )).to_numpy( allow_missing=True)
        masks['g'] = ak.flatten((matchedJets.slot0.partonFlavour==21)).to_numpy( allow_missing=True)

        # b_pos = (matchedJets.slot0.partonFlavour==5) | (matchedJets.slot0.partonFlavour==-5)
        # c_pos = (matchedJets.slot0.partonFlavour==4) | (matchedJets.slot0.partonFlavour==-4)
        # l_pos = ((matchedJets.slot0.partonFlavour==1) | (matchedJets.slot0.partonFlavour==-1) |
        #          (matchedJets.slot0.partonFlavour==2) | (matchedJets.slot0.partonFlavour==-2) |
        #          (matchedJets.slot0.partonFlavour==3) | (matchedJets.slot0.partonFlavour==-3) )        
        # g_pos = matchedJets.slot0.partonFlavour==21


        ptresponses            = { sample: ptresponse_np[masks[sample]]             for sample in subsamples }
        matched_genjetpts      = { sample: matched_genjetpt_np[masks[sample]]       for sample in subsamples }
        matched_genjetetas     = { sample: matched_genjeteta_np[masks[sample]]      for sample in subsamples }

        # for sample in sumsamples:
            # ptresponses
        
        # ptresponse_b = ptresponse[b_pos]
        # ptresponse_c = ptresponse[c_pos]
        # ptresponse_l = ptresponse[l_pos]
        # ptresponse_g = ptresponse[g_pos]



        output['jetpt'].fill(dataset = dataset, 
                        pt = ak.flatten(matched_genjetpt))

        output['jeteta'].fill(dataset = dataset, 
                        jeteta = ak.to_numpy(ak.flatten(jeteta), allow_missing=True))
       

 #         output['ptresponse'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(matched_genjetpt), allow_missing=True),
 #                             jeteta=ak.to_numpy(ak.flatten(matched_genjeteta), allow_missing=True),
 #                             ptresponse=ak.to_numpy(ak.flatten(ptresponse), allow_missing=True))
        
        output['ptresponse'].fill(dataset=dataset, pt=matched_genjetpt_np,
                                                   jeteta=matched_genjeteta_np,
                                                   ptresponse=ptresponse_np)
  
        for sample in subsamples:
               output['ptresponse_'+sample].fill(dataset=dataset, pt=matched_genjetpts[sample],
                                    jeteta=matched_genjetetas[sample],
                                    ptresponse=ptresponses[sample])

#         for sample in subsamples:
#                output['ptresponse_'+sample].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(matched_genjetpts[sample]), allow_missing=True),
#                         jeteta=ak.to_numpy(ak.flatten(matched_genjetetas[sample]), allow_missing=True),
#                         ptresponse=ak.to_numpy(ak.flatten(ptresponses[sample] ), allow_missing=True))
        
        return output

    def postprocess(self, accumulator):
        return accumulator



