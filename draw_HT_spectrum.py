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
        h_HT = hist.Hist(self.HT_gen_axis, name="Counts")
        h_HT_theory = hist.Hist(self.HT_gen_axis, name="Counts")
        h_HT.fill(events.LHE.HT, weight=events.LHEWeight.originalXWGTUP)
        h_HT_theory.fill(events.LHE.HT) #, weight=xsec_dict[key]);

        h_HT = h_HT/len(events)
        h_HT_theory = h_HT_theory/len(events)*events.metadata['xsec']

        print(f"mean LHEweight = {ak.mean(events.LHEWeight.originalXWGTUP)}, xsection = {events.metadata['xsec']}, len events = {len(events)}")
        return {events.metadata['dataset']: {"h_HT": h_HT, "h_HT_theory": h_HT_theory}}

    def postprocess(self, accumulator):
        return accumulator

