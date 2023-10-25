processor_dependencies = ['LHE_flavour.py', 'common_binning.py', 'JERCProcessorcuts.py']

processor_config = {
    # cut: {cut_parameters}
    "good_lepton_cut":      {"apply":True},
    "tight_lepton_veto_id": {"apply":True},
    "recolep_drcut": {"apply":True},
    "jet_pt_cut": {
        "apply":True,
        "mingenjetpt": 15,
    },
    "leading_jet_and_alpha_cut":{
        "apply":True,
        "alphaQCD": 1.0,
        "alphaDY":  1.0,
        "NjetsQCD": 3,
        "NjetsDY":  2,
    },
    "select_Nth_jet":{
        "apply":False,
        "N": 2,
    },
    "jet_iso_cut":{"apply":False,
                   "dr_cut": 0.8,
    },
    "reco_jetMCmatching":{"apply":True,
                          "dR":   0.2,
    },
    ### Choose the jet flavour. Some samples have missing `partonFlavour`, so one has to redo the flavour matching oneself. Two different option were implemented:
    ### `LHE_flavour` starts from the jet and matches to the closest LHE particle.
    ### `LHE_flavour2` (a better option) starts from the LHE particle and matches to the jet 
    "jetflavour":'partonFlavour',
    "use_gen_weights": False,
    "use_pu_weights": True,
    # ... Add more cuts and parameters as needed
}