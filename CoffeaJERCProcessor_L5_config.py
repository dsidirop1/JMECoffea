processor_dependencies = ['LHE_flavour.py', 'common_binning.py', 'JERCProcessorcuts.py']

processor_config = {
    # cut: {cut_parameters}
    "good_lepton_cut":      {"apply":True},
    "tight_lepton_veto_id": {"apply":True},
    "gen_vtx_dz_cut": {"apply": False},
    "recolep_drcut": {"apply":True},
    "jet_pt_cut": {
        "apply":True,
        "mingenjetpt": 15,
    },
    "leading_jet_and_alpha_cut":{
    # Alpha cut not used (alpha=1) since run 2 because the large pileup causes a bias
        "apply":True,
        "alphaQCD": 1.0,
        "alphaDY":  1.0,
        "NjetsQCD": 3, #-1 for all jets
        "NjetsDY":  2,
    },
    "select_Nth_jet":{
    # for debugging purposses, select exactly the Nth jet in each event
        "apply":False,
        "N": 2,
    },
    "jet_iso_cut":{"apply":True,
                   "dr_cut": 0.8,
    },
    "reco_jetMCmatching":{"apply":True,
                          "dR":   0.2,
    },
    ### Choose the jet flavour. Some samples have missing `partonFlavour`, so one has to redo the flavour matching oneself. Two different option were implemented:
    ### `LHE_flavour` starts from the jet and matches to the closest LHE particle.
    ### `LHE_flavour2` (a better option) starts from the LHE particle and matches to the jet. Was slower than LHE_flavor before a numba compiled implementation for ak.count was made. 
    "jetflavour":'partonFlavour',
    "use_gen_weights": False,
    "use_pu_weights": True,
    "split_ISR_FSR_gluons": False,  # based on if parton flavor overlaps with LHE flavor for some samples it can be possible to split FSR and ISR
    # ... Add more cuts and parameters as needed
}