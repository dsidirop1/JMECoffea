# JERCProcessorcuts.py

import awkward as ak
import numpy as np

### input numbers
mZpdg = 91.1876

def inv_mass_plus(lepton_pairs):
    ''' Compute the invariant mass of the leptons in `lepton pairs`
    '''
    return np.sqrt(np.abs(np.sum(lepton_pairs.E, axis=1)**2 - np.sum(lepton_pairs.px, axis=1)**2
                    - np.sum(lepton_pairs.py, axis=1)**2 - np.sum(lepton_pairs.pz, axis=1)**2))

def jet_iso_cut(reco_jets):
    ''' Jet isolation cut'''
    drs, _ = reco_jets.metric_table(reco_jets, return_combinations=True, axis=1)
    jet_iso_mask = ~ ak.any((1e-10<drs) & (drs<0.8), axis=2 )
    return reco_jets[jet_iso_mask]

def leading_jet_and_alpha_cut(reco_jets, leptons, events, dataset, alphaQCD, alphaDY, NjetsQCD, NjetsDY):
    ''' Alpha cut = cut on the additional jet activity.
    Not used (alpha=1) since run 2 because the large pileup causes a bias
    '''
    if "QCD" in dataset:
        alphacut = alphaQCD #if the alpha cut is different from the default
        # Correctly/safely treat the cases where there are less then 3 jets left after the cuts
        # select only the first three jets on QCD samples
        # to avoid effects due to a non-physical jet spectrum 
        if NjetsQCD>-1:
            reco_jetspt = ak.pad_none(reco_jets.pt, NjetsQCD, axis=1, clip=True)

        if NjetsQCD>2:
            alpha = reco_jetspt[:,2]*2/(reco_jetspt[:,0]+reco_jetspt[:,1])
            alpha = ak.fill_none(alpha,0)
            reco_jets = reco_jets[alpha<alphacut][:,:3]
            events = events[alpha<alphacut]
        
    elif 'DY' in dataset:
        alphacut = alphaDY #if the alpha cut is different from the default
        reco_jetspt = ak.pad_none(reco_jets.pt, NjetsDY, axis=1, clip=True)
        alpha = reco_jetspt[:,1]/ak.sum(leptons.pt,axis=1)
        alpha = ak.fill_none(alpha,0)
        reco_jets = reco_jets[alpha<alphacut][:,:2]
        events = events[alpha<alphacut]

    return reco_jets, events

def jet_pt_cut(reco_jets, mingenjetpt):
    jet_pt_mask = reco_jets.matched_gen.pt>mingenjetpt
    ## funny workaround to change the ak.type of jet_pt_mask from '10 * var * ?bool' to '10 * var * bool'
    ## otherwise after the correction .matched_gen field is not found.
    jet_pt_mask_shape = ak.num(jet_pt_mask)
    jet_pt_mask_np = ak.flatten(jet_pt_mask).to_numpy()
    jet_pt_mask = ak.unflatten(jet_pt_mask_np.data, jet_pt_mask_shape)
    reco_jets = reco_jets[jet_pt_mask]
    return reco_jets

def good_lepton_cut(reco_jets, events, dataset, leptons, tightelectrons, tightmuons):
    '''Comparing to number of generated prompt leptons can deal with all the sample cases (2 for DY, 0,1,2 for TTBAR, 0 for QCD)
    Cuts on DY and ttbar based on L3Res selections https://twiki.cern.ch/twiki/bin/view/CMS/L3ResZJet
    '''
    events_with_good_lep = ((ak.num(tightmuons) == ak.num(leptons))
                    | (ak.num(tightelectrons) == ak.num(leptons) )
                    )        

    DYcond = np.array([True]*len(events))

    if 'DY' in dataset:
        DYcond = DYcond * (
            (np.sum(tightelectrons.pt, axis=1)>15) | (np.sum(tightmuons.pt, axis=1)>15)
        )
        DYcond = DYcond * (
            (np.abs(inv_mass_plus(tightelectrons) - mZpdg) < 20)
            | (np.abs(inv_mass_plus(tightmuons) - mZpdg) < 20)
        )

    events = events[events_with_good_lep*DYcond]
    reco_jets = reco_jets[events_with_good_lep*DYcond]
    leptons = leptons[events_with_good_lep*DYcond]
    tightelectrons = tightelectrons[events_with_good_lep*DYcond]
    tightmuons = tightmuons[events_with_good_lep*DYcond]
    return(events, reco_jets, leptons, tightelectrons, tightmuons)

def select_leptons(selectedEvents):
    ''' Select leptons according to the L3Res selections https://twiki.cern.ch/twiki/bin/view/CMS/L3ResZJet
    '''
    muon = selectedEvents.Muon
    tight_mu_cut = (muon.tightId) & (muon.pfIsoId>=4) & (np.abs(muon.eta)<2.3) & (muon.pt>20)
    tightmuons = muon[tight_mu_cut]

    electron = selectedEvents.Electron
    tight_ele_cut = (electron.cutBased==4) &(np.abs(electron.eta)<2.4) & (electron.pt>25)
    tightelectrons = electron[tight_ele_cut]
    
    genpart = selectedEvents.GenPart
    lepton_mask = (
            ((np.abs(genpart.pdgId) == 11) | (np.abs(genpart.pdgId) == 13) | (np.abs(genpart.pdgId) == 15 ))
            & (genpart.statusFlags>>13&1 == 1) 
            & (genpart.statusFlags&1 == 1)
    )
    leptons = genpart[lepton_mask]
    return leptons, tightelectrons, tightmuons

def recolep_drcut(reco_jets, tightelectrons, tightmuons):
    ''' Additional dR cut on not overlapping with leptons
    (tight lepton veto id does not seem to cut all the leptons)
    '''
    drs = reco_jets.metric_table(tightelectrons, return_combinations=False, axis=1 )
    matched_with_promt_lep = np.any((drs<0.4),axis=2)
    overlappng_reco_lep_mask = np.logical_not(matched_with_promt_lep)

    drs = reco_jets.metric_table(tightmuons, return_combinations=False, axis=1 )
    matched_with_promt_lep = np.any((drs<0.4),axis=2)
    overlappng_reco_lep_mask = overlappng_reco_lep_mask*np.logical_not(matched_with_promt_lep)
    reco_jets = reco_jets[overlappng_reco_lep_mask]
    return reco_jets

def wrong_recolep_drcut(reco_jets, cut_prompt_lep):
    ''' Additional dR cut on not overlapping with leptons
    cut_prompt_lep = True: cut on only prompt leptons, else cut on all leptons (including from semileptonic jet decays -> wrong)
    '''
    if cut_prompt_lep:
        ele_partFlav = reco_jets.matched_electrons.genPartFlav
        mu_partFlav = reco_jets.matched_muons.genPartFlav
        dressed_electron_mask = np.logical_not(np.sum((ele_partFlav == 1) | (ele_partFlav == 15),axis=2))
        dressed_muon_mask = np.logical_not(np.sum((mu_partFlav == 1) | (mu_partFlav == 15),axis=2))
    else:
        dressed_electron_mask = ak.sum(ak.is_none(reco_jets.matched_electrons,axis=2), axis=2)==2
        dressed_muon_mask     = ak.sum(ak.is_none(reco_jets.matched_muons,axis=2), axis=2)==2

    jet_mask = dressed_electron_mask & dressed_muon_mask
    return reco_jets[jet_mask]

def select_Nth_jet(reco_jets, selectedEvents, N):
    ''' Select the Nth jet.
    For example, when N=3: take the jet that comes from ME in MG+Pythia8 and from the shower in Pythia8
    '''
    N_jets_exist = ak.num(reco_jets)>=N
    reco_jets = reco_jets[N_jets_exist]
    selectedEvents = selectedEvents[N_jets_exist]
    reco_jets = reco_jets[:,N-1:N]
    return reco_jets, selectedEvents