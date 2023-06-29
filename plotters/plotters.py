import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mplhep as hep
import os
from common_binning import JERC_Constants

def plot_corrections(mean, meanstd, ptbins_c, etabins, tag, flavor, plotetavals=[0, 1.305, 2.5, 3.139]):

    ### To ignore the points with 0 on y axis when setting the y axis limits
    mean_p = mean.copy()
    mean_p[mean_p==0] = np.nan

    fig, ax = plt.subplots() 
    start = np.searchsorted(ptbins_c, 20, side='left') #np.where(ptbins<=20)[0][-1]
    
    etaidxs = etabins.get_bin_idx(plotetavals)
#     lastbin = lastbin[-1] if len(lastbin)>0 else -1
    lastbin = -1
    
    ptbins_plot = ptbins_c[start:lastbin]
    meanstd = meanstd[start:lastbin,:]
    mean_p = mean_p[start:lastbin]
    
#     print(f'ptvinsplot = {ptbins_plot}')
    for etaidx in etaidxs:
        plt.errorbar(ptbins_plot, mean_p[:,etaidx], yerr=meanstd[:,etaidx], marker='o',
                     linestyle="none", label=etabins.idx2plot_str(etaidx),
                     capsize=1.6, capthick=0.7, linewidth=1.0)

    ### Calculate resonable limits excluding the few points with insane errors
    yerr_norm = np.concatenate([meanstd[:,etaidxs] ])
    y_norm = np.concatenate([mean_p[:,etaidxs]])
    norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
    
    # if for too low statistics that no point passes the criteria then plot full y-range
    if np.sum(norm_pos!=0):
        left_lim = np.min((y_norm-yerr_norm)[norm_pos])
        right_lim = np.max((yerr_norm+y_norm)[norm_pos])
        lim_pad = (right_lim - left_lim)/20
        ax.set_ylim(left_lim-lim_pad, right_lim+lim_pad)
    
    ax.set_xscale('log')
    
    good_xlims = ax.get_xlim()
    ax.set_xticks([20, 50, 100, 500, 1000, 5000])
    ax.set_xlim(good_xlims)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel(r'$p_{T,Gen}$ (GeV)');
    ax.set_ylabel(r'median response');
    ax.legend()

    tag = tag[4:] #clip the _L5_
    FitFigDir1 = 'fig/corr_vs_pt'
    FitFigDir2 = FitFigDir1+'/'+tag
    if not os.path.exists(FitFigDir1):
        os.mkdir(FitFigDir1)
    if not os.path.exists(FitFigDir2):
        os.mkdir(FitFigDir2)

    hep.label.exp_text(text=f'{flavor} jets, {tag} sample', loc=0, ax=ax)
    figname = FitFigDir2+'/corr_vs_pt_'+tag+'_'+flavor
    plt.savefig(figname+'.pdf');
    plt.savefig(figname+'.png');
    print(f'Figure saved: {figname}.pdf /.png')

    plt.show();


def plot_corrections_eta(mean, meanstd, ptbins, etabins_c, tag, flavor, plotptvals=[15, 35, 150, 400]):
    ### To ignore the points with 0 on y axis when setting the y axis limits
    mean_p = mean.copy()
    mean_p[mean_p==0] = np.nan

    fig, ax = plt.subplots()
#     start = np.where(ptbins<=20)[0][-1]
    
#     ptbins_plot = ptbins_c[start:]
#     meanstd = meanstd[start:,:]
#     mean_p = mean_p[start:]
    
    ptidxs = ptbins.get_bin_idx(plotptvals)    
#     np.isnan(mean_p[k2,:]*mean_p[k4,:]*mean_p[k6,:]*mean_p[k8,:])
    
    for ptidx in ptidxs:
        plt.errorbar(etabins_c, mean_p[ptidx,:], yerr=meanstd[ptidx], marker='o',
                linestyle="none", label=ptbins.idx2plot_str(ptidx),
                capsize=1.6, capthick=0.7, linewidth=1.0)

    for HCal_border in JERC_Constants.etaBinsEdges_Win14():
        ax.vlines(HCal_border,0, 2, linestyles='--',color="gray",
            linewidth=1,)

    ### Calculate resonable limits excluding the few points with insane errors
    yerr_norm = np.concatenate([meanstd[ptidxs] ])
    y_norm = np.concatenate([mean_p[ptidxs]])
    norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
    ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]) ,np.max((yerr_norm+y_norm)[norm_pos]))
    ax.set_xlabel(r'$|\eta|$');
    ax.set_ylabel(r'median response');
#     ax.set_xscale('log')
    ax.legend()
    
    tag = tag[4:]
    FitFigDir1 = 'fig/corr_vs_pt'
    FitFigDir2 = FitFigDir1+'/'+tag
    if not os.path.exists(FitFigDir1):
        os.mkdir(FitFigDir1)
    if not os.path.exists(FitFigDir2):
        os.mkdir(FitFigDir2)

    hep.label.exp_text(text=f'{flavor} jets, {tag} sample', loc=0, ax=ax)
    figname = FitFigDir2+'/corr_vs_pt_'+tag+'_'+flavor
    plt.savefig(figname+'.pdf');
    plt.savefig(figname+'.png');
    print(f'Figure saved: {figname}.pdf /.png')
    plt.show();
    
from helpers import gauss
def plot_response_dist(histo, xvals, p2, cov, chi2, Ndof, median, medianstd, Neff, figName, hep_txt='' ):
    width_ik = np.abs(p2[2])
    f_xvals = np.linspace(0,max(xvals),5001)
    fgaus2 = gauss(f_xvals, *p2)
    edd = histo.axes['ptresponse'].edges
    histo = histo[:len(edd)-2] 
    
    fig, ax2 = plt.subplots();
    plot = histo.plot1d(ax=ax2, label='dataset', histtype='fill', alpha=0.6)
    ax2.plot(f_xvals, fgaus2, label='Gaus', markersize=0, linewidth=1.8)
    ax2.set_xlabel("Response ($p_{T,reco}/p_{T,ptcl}$)")
    max_lim = np.min([np.max(xvals), 2.0])
    ax2.set_xlim([0,max_lim])
    h = ax2.get_ylim()[1]/1.05
    plt.text(0.03,0.95*h,r'Mean {0:0.3f}$\pm${1:0.3f}'.format(p2[1], np.sqrt(cov[1,1])))
    plt.text(0.03,0.88*h,r'Width {0:0.3f}$\pm${1:0.3f}'.format(width_ik, np.sqrt(cov[2,2])))
    plt.text(0.03,0.81*h,r'Median {0:0.3f}$\pm${1:0.3f}'.format(median, medianstd))
    plt.text(0.03,0.73*h,r'$\chi^2/ndof$ {0:0.2g}/{1:0.0f}'.format(chi2, Ndof))
    plt.text(0.03,0.66*h,r'Neff = {0:0.3g}'.format(Neff))
    ax2.legend();
    
    hep.label.exp_text(text=hep_txt, loc=0)
    plt.savefig(figName+'.png', dpi=plt.rcParamsDefault['figure.dpi']);
    plt.savefig(figName+'.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
    plt.show();   