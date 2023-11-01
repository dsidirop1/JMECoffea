legend_labels = {"ttbar": {"lab":"$t\overline{\, t\!}$ ",
                            "short": "ttbar"},
                "QCD": {"lab":"QCD",
                        "short": "QCD"},
                "DY": {"lab":"DY",
                        "short": "DY"}
                            }

'''
Lift of available datasets.

Datasets can either given by a path to a file storing file names or by a path to a file storing datasets
and their cross-sections

data_tag: [file_name_path, xsec_path, legend_label]
'''
dataset_dictionary = {
    "Pythia-TTBAR": [None, 'fileNames/TTBAR_Pythia_20UL18/xsecs_TTBAR_Pow-Py8.txt', legend_labels["ttbar"]["lab"]+'Pow+Py8'],
    "Pythia-semilep-TTBAR": ['fileNames/TTBAR_Pythia_20UL18/TTToSemi20UL18_JMENano.txt', 1, legend_labels["ttbar"]["lab"]+'Pow+Py8'],
    "Pythia-non-semilep-TTBAR": [None, 'fileNames/TTBAR_Pythia_20UL18/xsecs_TTBAR_Pow-Py8-non-semilep.txt', legend_labels["ttbar"]["lab"]+'Pow+Py8'],
    "Herwig-TTBAR": ['fileNames/TT20UL18_JMENano_Herwig.txt', 1, legend_labels["ttbar"]["lab"]+'Pow+Her7'],
    "DY-MG-Py":     ['fileNames/DYJets_MG-Py.txt', 1, 'ZJets MG+Py8'],
    "DY-MG-Her":    ['fileNames/DYJets_MG-Her.txt', 1, 'ZJets MG+Her7'],
    "QCD-MG-Py":    [None, 'fileNames/QCD_MG_Py8_20UL18/xsecs_QCD_MG_py8.txt', 'QCD MG+Py8'],
    # "QCD-MG-Her":   [None, 'fileNames/QCD_MG_Py8_20UL18/xsecs_QCD_MG_py8.txt', 'QCD MG+Her7'],
    "QCD-MG-Her":   [None, 'fileNames/QCD_Herwig_20UL18/xsecs_QCD_Herwig_corrected.txt', 'QCD MG+Her7'],
    "QCD-Py":       ['fileNames/QCD20UL18_JMENano.txt', 1, 'QCD Py8'],
    "DY-FxFx":      ['fileNames/DYJets.txt', 1, 'ZJets FxFx'],
    "noJME-QCD-Py_pu": ['fileNames/fileNames_QCD20UL18.txt', 1, 'QCD Py8'],
    "scaled_pion_kaon": ['fileNames_pion_response/fileNames_scaled_pion_kaon.txt', 1, 'pion up/ kaon up'],
#     "scaled_pion": ['fileNames_pion_response/fileNames_scaled_pion.txt', 1, 'pion up'],
    "scaled_pion": ['fileNames_pion_response/fileNames_scaled_times1_pion.txt', 1, 'pion up'],
    "scaled_times2_pion": ['fileNames_pion_response/fileNames_scaled_times2_pion.txt', 1, 'pion up'],
    "scaled_times5_pion": ['fileNames_pion_response/fileNames_scaled_times5_pion.txt', 1, 'pion up'],
    "scaled_times10_pion": ['fileNames_pion_response/fileNames_scaled_times10_pion.txt', 1, 'pion up'],
    "scaled_times100_pion": ['fileNames_pion_response/fileNames_scaled_times100_pion.txt', 1, 'pion up'],
    "not_scaled_pion": ['fileNames_pion_response/fileNames_not_scaled_pion.txt', 1, 'pion central'],}