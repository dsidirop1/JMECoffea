### File borrowed from
### https://gitlab.cern.ch/cms-jetmet/JERCProtoLab/-/blob/master/macros/common_info/common_binning.py
### but updated but many new binnings used/tried in MC truth flavor analysis.

import parse
import numpy as np

def mirrorMiPlbins(bins):
    return np.array(list((np.array(bins)*(-1) )[-1:0:-1] ) + list(bins))


class JERC_Constants():

    @staticmethod
    def etaBinsEdges_CaloTowers():
        return [0.000, 0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.879, 0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830, 1.930, 2.043, 2.172, 2.322, 2.500, 2.650, 2.853, 2.964, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191]

    @staticmethod
    def etaBinsEdges_CaloTowers_full():
        return mirrorMiPlbins(JERC_Constants.etaBinsEdges_CaloTowers())

    @staticmethod
    def etaBinsEdges_Win14():
        return [0., 1.305, 2.5, 3.139, 5.191]

    @staticmethod
    def etaBinsEdges_Win14_full():
        return mirrorMiPlbins(JERC_Constants.etaBinsEdges_Win14())

    @staticmethod
    def etaBinsEdges_onebin():
        return [0., 5.191]

    @staticmethod
    def etaBinsEdges_onebin_full():
        return [-5.191, 5.191]

    @staticmethod
    def etaBinsEdges_Aut18():
        return [0,  0.783,  1.305,  1.653,   1.93,  2.322,    2.5,  2.853,  3.139,  3.489, 5.191]

    @staticmethod
    def etaBinsEdges_Aut18_full():
        return mirrorMiPlbins(JERC_Constants.etaBinsEdges_Aut18())

    @staticmethod
    def etaBinsEdges_JERC():
        return [0.000, 0.261, 0.522, 0.783, 1.044, 1.305, 1.566, 1.740, 1.930, 2.043, 2.172, 2.322, 2.500, 2.650, 2.853, 2.964, 3.139, 3.489, 3.839, 5.191]

    @staticmethod
    def etaBinsEdges_JERC_full():
        return mirrorMiPlbins(JERC_Constants.etaBinsEdges_JERC())

    @staticmethod
    def etaBinsEdges_Summer20Flavor():
        return [0.000, 0.261, 0.522, 0.783, 1.044, 1.305, 1.566, 1.740, 1.930, 2.043, 2.172, 2.500, 2.964, 5.191]

    @staticmethod
    def etaBinsEdges_Summer20Flavor_full():
        return mirrorMiPlbins(JERC_Constants.etaBinsEdges_Summer20Flavor())

    @staticmethod
    def ptBinsEdgesMCTruth():
        return [15.0, 17.0, 20.0, 23.0, 27.0, 30.0, 35.0, 40.0, 45.0, 57.0, 72.0, 90.0, 120.0, 150.0, 200.0, 300.0, 400.0, 550.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 10000]

    # @staticmethod
    # def ptBinsEdgesMCTruth17():
    #     return [17.0, 20.0, 23.0, 27.0, 30.0, 35.0, 40.0, 45.0, 57.0, 72.0, 90.0, 120.0, 150.0, 200.0, 300.0, 400.0, 550.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0]

    @staticmethod
    def etaBinsEdges_Uncert():
        return np.array([0. ,  0.2, 0.4,  0.6,  0.8,  1. ,  1.2,  1.4,  1.6,  1.8,  2. ,  2.2,  2.4, 2.6,  2.8,  3. ,  3.5,  4. ,  4.4,  5. ,  5.4])

    @staticmethod
    def etaBinsEdges_Uncert_full():
        return mirrorMiPlbins(JERC_Constants.etaBinsEdges_Uncert())
    
    @staticmethod
    def ptBinsEdges_Uncert():
        return np.array([   9. ,   11. ,   13.5,   16.5,   19.5,   22.5,   26. ,   30. ,
                        34.5,   40. ,   46. ,   52.5,   60. ,   69. ,   79. ,   90.5,
                        105.5,  123.5,  143. ,  163.5,  185. ,  208. ,  232.5,  258.5,
                        286. ,  331. ,  396. ,  468.5,  549.5,  639. ,  738. ,  847.5,
                        968.5, 1102. , 1249.5, 1412. , 1590.5, 1787. , 2003. , 2241. ,
                    2503. , 2790.5, 3107. , 3455. , 3837. , 4257. , 4719. , 5226.5,
                    5784. , 6538. ])


    @staticmethod
    def StrToBinsDict(absolute=True):
        if absolute==True:
            return {"HCalPart": JERC_Constants().etaBinsEdges_Win14(),
                "CoarseCalo": JERC_Constants().etaBinsEdges_Aut18(),
                "JERC" : JERC_Constants().etaBinsEdges_JERC(),
                "CaloTowers": JERC_Constants().etaBinsEdges_CaloTowers(),
                "one_bin": JERC_Constants().etaBinsEdges_onebin(),
                "Uncert": JERC_Constants().etaBinsEdges_Uncert(),
                "Summer20Flavor": JERC_Constants().etaBinsEdges_Summer20Flavor(),
                }
        else:
            return {
                "HCalPart": JERC_Constants().etaBinsEdges_Win14_full(),
                "CoarseCalo": JERC_Constants().etaBinsEdges_Aut18_full(),
                "JERC" : JERC_Constants().etaBinsEdges_JERC_full(),
                "CaloTowers": JERC_Constants().etaBinsEdges_CaloTowers_full(),
                "one_bin": JERC_Constants().etaBinsEdges_onebin_full(),
                "Uncert": JERC_Constants().etaBinsEdges_Uncert_full(),
                "Summer20Flavor": JERC_Constants().etaBinsEdges_Summer20Flavor_full(),
                }
    
    def StrToPtBinsDict():
       return {"MC_truth": JERC_Constants().ptBinsEdgesMCTruth(),
               # "MC_truth17": JERC_Constants().ptBinsEdgesMCTruth17(),
               "Uncert": JERC_Constants().ptBinsEdges_Uncert()}

    @staticmethod
    def str2bins(bin_string, absolute=True):
        return JERC_Constants.StrToBinsDict(absolute=absolute)[bin_string]

    @staticmethod
    def BinToString(low, high, precision=3):
         mystr = '{:.{}f}'.format(low,precision)
         mystr += '_'
         mystr += '{:.{}f}'.format(high,precision)
         mystr = mystr.replace('.','p')
         return mystr

    @staticmethod
    def StringToBin(mystr, precision=3):
        bins = parse.compile('{low}_{high}').parse(mystr).named
        bins['low'] = float(bins['low'].replace('p','.'))
        bins['high'] = float(bins['high'].replace('p','.'))
        return (bins['low'],bins['high'])

    @staticmethod
    def GetEtaBinEdgeMin(val, binning=None):
        if binning == None:
            binning = JERC_Constants().etaBinsEdges_JERC()
        
        return round(list(filter(lambda x: x<=val, JERC_Constants().str2bins(binning)))[-1],4)

    @staticmethod
    def GetEtaBinEdgeMax(val, binning=None):
        if binning == None:
            binning = JERC_Constants().etaBinsEdges_JERC()
        return round(list(filter(lambda x: x>val, JERC_Constants().str2bins(binning)))[0],4)

    @staticmethod
    def GetEtaBinCenter(val, binning=None):
        if binning == None:
            binning = JERC_Constants().etaBinsEdges_JERC()
        min = JERC_Constants().GetEtaBinEdgeMin(val, JERC_Constants().str2bins(binning))
        max = JERC_Constants().GetEtaBinEdgeMax(val, JERC_Constants().str2bins(binning))
        return round((max+min)/2,4)

    @staticmethod
    def GetEtaBinWidth(val, binning=None):
        if binning == None:
            binning = JERC_Constants().etaBinsEdges_JERC()
        min = JERC_Constants().GetEtaBinEdgeMin(val, JERC_Constants().str2bins(binning))
        max = JERC_Constants().GetEtaBinEdgeMax(val, JERC_Constants().str2bins(binning))
        return round((max-min)/2,4)
    
    # @staticmethod
    # def GetEtaBinIdx(vals, binning_str=None):
    #     if not type(vals) is np.ndarray:
    #         vals = np.array(vals)
    #     if len(vals.shape)==0:
    #         vals = np.array([vals])

    #     if binning_str == None:
    #         binning = JERC_Constants().etaBinsEdges_JERC()
    #     else:
    #         binning = JERC_Constants().str2bins(binning_str)
            
    #     eta_idxs = np.clip(np.searchsorted(binning, vals, "right")-1, 0, len(binning)-2)
    #     return eta_idxs
