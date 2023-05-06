# LHE_flavour.py
import awkward as ak
import numpy as np
from count_2d import count_2d


def get_LHE_flavour_2(jets, selectedEvents):
    ''' Algorithm of LHE_Flavour2 for the jet:
        Cuts all the outgoing LHE particles that have pdgId as quarks (except top) and gluons.
        For each LHE particle finds the closest jet and gives the jet its flavour.
        If a jet is marked by two or more LHE particles: assign -999
        Using the numba compiled `count_2d` funciton.
        '''
    
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

    return jets

def get_LHE_flavour(reco_jets, selectedEvents):
    """ Algorithm for the flavour derivation:
    - Find all the matched outgoing LHE particles within dR<0.4
    - If there is at least one LHE particle with b flavour (bbar flavour), set LHE_flavour to 5 (-5). If both b and bbar are found, set LHE_flavour=0
    - If there is no b quark then: 
    If there is at least one LHE particle with c flavour (cbar flavour), set LHE_flavour to 4 (-4).
    If both are found, set LHE_flavour=0.
    - If none of the above:
    Assign the flavour of the hardest selected LHE particle.
    """
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
    return reco_jets