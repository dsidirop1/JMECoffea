### A super simple processor for adding up the HT bins.

# workaround to get a locally installed coffea and awkwrd version using lch on lxplus
# comment out or replace the path if I happened to forget to remove these lines before pushing:
import sys
sys.path.insert(0,'/afs/cern.ch/user/a/anpotreb/top/JERC/coffea')
ak_path = '/afs/cern.ch/user/a/anpotreb/top/JERC/local-packages/'
if ak_path not in sys.path:
        sys.path.insert(0,ak_path)

import hist
import awkward as ak
# from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor

class Processor(processor.ProcessorABC):
    def __init__(self, processor_config):
        self.HT_gen_axis = hist.axis.Regular(600, 50, 3050, overflow=True, underflow=True, name="$HT$")

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events): 
        if 'LHEWeight' not in events.fields: ### no LHEWeight.originalXWGTUP stored in standalone Pythia8 but Generator.weight instead
            weights = events.Generator.weight
        else:
            weights = events.LHEWeight.originalXWGTUP

        cutflow_axis = hist.axis.StrCategory([], growth=True, name="cutflow", label="Cutflow Scenarios")
        cutflow = hist.Hist(cutflow_axis, storage="weight", label="Counts")
    
        ############ Event Cuts ############
        # apply npv cuts
        cutflow.fill(cutflow='all_events', weight=len(events))

        h_HT = hist.Hist(self.HT_gen_axis, name="Counts", storage="weight")
        h_HT_theory = hist.Hist(self.HT_gen_axis, name="Counts", storage="weight") 
        h_HT.fill(events.LHE.HT, weight=weights)                    ### Using the generated weights which do not stitch well together
        h_HT_theory.fill(events.LHE.HT) #, weight=xsec_dict[key]);  ### Using the xsecs from the database that for Herwig need to be adjusted anyway

        h_HT = h_HT               #/len(events)
        h_HT_theory = h_HT_theory #/len(events)*events.metadata['xsec']
            
        # print(f"mean LHEweight = {ak.mean(weights)}, xsection = {events.metadata['xsec']}, len events = {len(events)}")
        return {events.metadata['dataset']: {"h_HT": h_HT, "h_HT_theory": h_HT_theory, "cutflow": cutflow}}

    def postprocess(self, accumulator):
        return accumulator

