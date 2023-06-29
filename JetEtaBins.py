#JetEtaBins.py
from common_binning import JERC_Constants
from dataclasses import dataclass, field
from collections.abc import Iterable
import numpy as np

@dataclass()
class JetBins():
    bin_type: str
    edges:   np.ndarray = field(init=False, repr=True)
    centres: np.ndarray = field(init=False, repr=True)
    nbins:   np.ndarray = field(init=False, repr=True)


    # def __post_init__(self):
    #     bin_dict = JERC_Constants.StrToBinsDict()
    #     if not self.bin_type in bin_dict.keys():
    #         raise ValueError(f"The eta bin type not in available binnings. Available binnings: {bin_dict.keys()}. The key given {self.bin_type}")
    #     self.edges = np.array(bin_dict[self.bin_type])
    #     self.centres = (self.edges[:-1] + self.edges[1:])/2
    #     self.nbins = len(self.centres)

    def idx2plot_str(self, idx, precision=3, var_name='|\eta|', dimension=''):
        if len(dimension)>0:
            dimension = ' '+dimension
        if isinstance(idx, Iterable):
            idx = idx[0]
        outstr = f'$ {np.round(self.edges[idx],precision)}<{var_name}<{np.round(self.edges[idx+1],precision)}${dimension}'
        return outstr

    def idx2str(self, idx, precision=3, var_name='eta'):
        if isinstance(idx, Iterable):
            idx = idx[0]
        outstr = f'{var_name}{np.round(self.edges[idx],precision)}to{np.round(self.edges[idx+1],precision)}'
        if precision>0:
            outstr = outstr.replace('.','p')
        else:
            outstr = outstr.replace('.0','')
        return outstr
    
    def get_bin_idx(self, etavals):
        # if not type(etavals) is np.ndarray:
        #     etavals = np.array(etavals)
        # if len(etavals.shape)==0:
        #     etavals = np.array([etavals])
        indx = np.searchsorted(self.edges, etavals, side='right')-1
        return np.clip(indx,0,len(self.edges)-2)
    
    def set_centres(self, centres: np.ndarray):
        if not len(centres)==self.nbins:
            raise ValueError(f"The length of given centres values is not equal to the number of bins, nbins = {self.nbins}")

        self.centres = np.array(centres)

@dataclass()
class JetEtaBins(JetBins):
    bin_type: str = "HCalPart"
    absolute: bool = False
    edges:   np.ndarray = field(init=False, repr=True)
    centres: np.ndarray = field(init=False, repr=True)
    nbins:   np.ndarray = field(init=False, repr=True)


    def __post_init__(self):
        bin_dict = JERC_Constants.StrToBinsDict(absolute=self.absolute)
        if not self.bin_type in bin_dict.keys():
            raise ValueError(f"The eta bin type not in available binnings. Available binnings: {bin_dict.keys()}. The key given {self.bin_type}")
        self.edges = np.array(bin_dict[self.bin_type])
        self.centres = (self.edges[:-1] + self.edges[1:])/2
        self.nbins = len(self.centres)

    def idx2plot_str(self, idx, precision=3):
        return super(JetEtaBins, self).idx2plot_str(idx, precision=precision, var_name='|\eta|')

    def idx2str(self, idx, precision=3):
        return super(JetEtaBins, self).idx2str(idx, precision=precision, var_name='eta')

@dataclass()
class PtBins(JetBins):
    bin_type: str = "MC_truth"
    edges:   np.ndarray = field(init=False, repr=True)
    centres: np.ndarray = field(init=False, repr=True)
    nbins:   np.ndarray = field(init=False, repr=True)


    def __post_init__(self):
        bin_dict = JERC_Constants.StrToPtBinsDict()
        if not self.bin_type in bin_dict.keys():
            raise ValueError(f"The pt bin type not in available binnings. Available binnings: {bin_dict.keys()}. The key given {self.bin_type}")
        if self.bin_type=='Uncert':
            self.centres = np.array(bin_dict[self.bin_type])
        else:
            self.edges = np.array(bin_dict[self.bin_type])
            self.centres = (self.edges[:-1] + self.edges[1:])/2
        self.nbins = len(self.centres)

    def idx2plot_str(self, idx, precision=0):
        return super(PtBins, self).idx2plot_str(idx, precision=precision, var_name='p_T', dimension='GeV')

    def idx2str(self, idx, precision=0):
        return super(PtBins, self).idx2str(idx, precision=precision, var_name='pt')




