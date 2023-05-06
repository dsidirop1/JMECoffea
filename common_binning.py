### File borrowed from
### https://gitlab.cern.ch/cms-jetmet/JERCProtoLab/-/blob/master/macros/common_info/common_binning.py

import parse
import numpy as np

def mirrorMiPlbins(bins):
    return list((np.array(bins)*(-1) )[-1:0:-1] ) + bins


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
    def etaBinsEdges_Aut18():
        return [0,  0.783,  1.305,  1.653,   1.93,  2.322,    2.5,  2.853,  3.139,  3.489, 5.191]

    @staticmethod
    def etaBinsEdges_Aut18_full():
        return mirrorMiPlbins(JERC_Constants.etaBinsEdges_Aut18())

    @staticmethod
    def etaBinsEdges_JERC():
        return [0.000, 0.261, 0.522, 0.783, 1.044, 1.305, 1.566, 1.740, 1.930, 2.043, 2.172, 2.322, 2.500, 2.650, 2.853, 2.964, 3.139, 3.489, 3.839, 5.191]


    @staticmethod
    def ptBinsEdgesMCTruth():
        return [15.0, 17.0, 20.0, 23.0, 27.0, 30.0, 35.0, 40.0, 45.0, 57.0, 72.0, 90.0, 120.0, 150.0, 200.0, 300.0, 400.0, 550.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0]

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
        return round(list(filter(lambda x: x<=val, binning))[-1],4)

    @staticmethod
    def GetEtaBinEdgeMax(val, binning=None):
        if binning == None:
            binning = JERC_Constants().etaBinsEdges_JERC()
        return round(list(filter(lambda x: x>val, binning))[0],4)

    @staticmethod
    def GetEtaBinCenter(val, binning=None):
        if binning == None:
            binning = JERC_Constants().etaBinsEdges_JERC()
        min = JERC_Constants().GetEtaBinEdgeMin(val, binning)
        max = JERC_Constants().GetEtaBinEdgeMax(val, binning)
        return round((max+min)/2,4)

    @staticmethod
    def GetEtaBinWidth(val, binning=None):
        if binning == None:
            binning = JERC_Constants().etaBinsEdges_JERC()
        min = JERC_Constants().GetEtaBinEdgeMin(val, binning)
        max = JERC_Constants().GetEtaBinEdgeMax(val, binning)
        return round((max-min)/2,4)
