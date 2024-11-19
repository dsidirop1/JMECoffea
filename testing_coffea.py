# test_opening_file.py
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import hist
from hist import Hist
import matplotlib.pyplot as plt

events = NanoEventsFactory.from_root(
   'root://xrootd-cms.infn.it//store/mc/RunIISummer20UL18NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/230000/9D0C102A-1A88-7D48-80A7-509AB9EAFD26.root',
schemaclass=NanoAODSchema.v6 ).events()

generator_particle = events.GenPart
print("show what variables are inside the 'GenPart'")
print(dir(generator_particle))
print('\n')
pt_values = []
print("show the variables 'pt', 'eta', 'phi', 'mass', 'pdgId' for the first 10 generated particles of the first event")
print('[pt,\t eta,\t phi,\t mass,\t pdgId]]')
for gen in generator_particle[:10,:]:
    #print(f'{gen.pt}, \t {gen.eta},\t {gen.phi},\t {gen.mass},\t {gen.pdgId}')
    pt_values.append(gen.pt)

h1 = plt.hist(pt_values)
plt.show()
plt.savefig("pt.png")

    
