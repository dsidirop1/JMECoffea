import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mplhep as hep
import os
import sys
# top_path = '../'
# if top_path not in sys.path:
#     sys.path.append(top_path)
from common_binning import JERC_Constants
from helpers import legend_str_to_filename

def plot_corrections(result, ptbins_c, etabins, tag, flavor, plotetavals=[0, 1.305, 2.5, 3.139], plotmean=True):
    ''' plotmean: if True plot not only median but also mean
    
    '''
    ### To ignore the points with 0 on y axis when setting the y axis limits
    # mean, meanstd, median, medianstd =  [{"a":2, "b":3,"3":5}[key] for key in ["a", "b"]] 
    median_p = result["Median"].copy()
    median_p[median_p==0] = np.nan
    if plotmean:
        mean_p = result["Mean"].copy()
        mean_p[mean_p==0] = np.nan

    fig, ax = plt.subplots() 
    start = np.searchsorted(ptbins_c, 20, side='left') #np.where(ptbins<=20)[0][-1]
    
    etaidxs = etabins.get_bin_idx(plotetavals)
#     lastbin = lastbin[-1] if len(lastbin)>0 else -1
    lastbin = -1
    
    ptbins_plot = ptbins_c[start:lastbin]
    medianstd = result["MedianStd"][start:lastbin,:]
    median_p  = median_p[start:lastbin]
    mean_p    = mean_p[start:lastbin] if plotmean else None
    meanstd   = result["MeanStd"][start:lastbin,:] if plotmean else None
    
#     print(f'ptvinsplot = {ptbins_plot}')
    points_ls = []
    for etaidx in etaidxs:
        mc = next(ax._get_lines.prop_cycler)
        points, _, _ = plt.errorbar(ptbins_plot, median_p[:,etaidx], yerr=medianstd[:,etaidx], # marker='o',
                     linestyle="none", label=etabins.idx2plot_str(etaidx),
                     capsize=1.6, capthick=0.7, linewidth=1.0, **mc)
        if plotmean:
            points2, caps, _ = plt.errorbar(ptbins_plot, mean_p[:,etaidx], yerr=meanstd[:,etaidx], # marker='o',
                     linestyle="none",
                     capsize=1.1, capthick=0.7, linewidth=1.0, mfc="white", markeredgewidth=1.2, **mc)
            
            caps[0].set_marker(".")
            caps[1].set_marker(".")
            if len(points_ls) == 0:
                points_ls.append(points)
                points_ls.append(points2)

    ### Calculate resonable limits excluding the few points with insane errors
    yerr_norm = np.concatenate([medianstd[:,etaidxs] ])
    y_norm = np.concatenate([median_p[:,etaidxs]])
    norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
    
    # if for too low statistics that no point passes the criteria then plot full y-range
    if np.sum(norm_pos!=0):
        left_lim = np.min((y_norm-yerr_norm)[norm_pos])
        right_lim = np.max((yerr_norm+y_norm)[norm_pos])
        lim_pad = (right_lim - left_lim)/20
        ax.set_ylim(left_lim-lim_pad, right_lim+lim_pad*13)
    
    ax.set_xscale('log')
    
    good_xlims = ax.get_xlim()
    ax.hlines(1,1, 10000, linestyles='--',color="black",
        linewidth=.9,)
    if flavor == 'all':
        ax.hlines(0.99,1, 10000, linestyles='dotted',color="black",
            linewidth=.9,)
        ax.hlines(1.01,1, 10000, linestyles='dotted',color="black",
            linewidth=.9,)
        ax.hlines(0.999,1, 10000, linestyles='dotted',color="black",
            linewidth=.9,)
        ax.hlines(1.001,1, 10000, linestyles='dotted',color="black",
            linewidth=0.9,)
    ax.set_xticks([20, 50, 100, 500, 1000, 5000])
    ax.set_xlim(good_xlims)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel(r'$p_{T,ptcl}$ (GeV)');
    ax.set_ylabel(r'median response');

    legend1 = ax.legend(points_ls, ["Median", "Mean"], loc="upper left", bbox_to_anchor=(0, 0.86))
    ax.legend(loc="upper right")
    if plotmean:
        ax.add_artist(legend1)

    tag = tag[4:] #clip the _L5_
    FitFigDir1 = 'fig/corr_vs_pt'
    FitFigDir2 = FitFigDir1+'/'+tag
    FitFigDir2 = legend_str_to_filename(FitFigDir2)
    if not os.path.exists(FitFigDir1):
        os.mkdir(FitFigDir1)
    if not os.path.exists(FitFigDir2):
        os.mkdir(FitFigDir2)

    hep.cms.label("Private work", loc=0, data=False, ax=ax, rlabel='')
    hep.label.exp_text(text=f'{flavor} jets\n{tag} sample', loc=2, ax=ax)
    figname = FitFigDir2+'/corr_vs_pt_'+tag+'_'+flavor
    figname = legend_str_to_filename(figname)
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
 
    good_xlims = ax.get_xlim()
    ax.hlines(1,-6, 6, linestyles='--',color="black",
        linewidth=.9,)
    ax.set_xlim(good_xlims)
 
    ### Calculate resonable limits excluding the few points with insane errors
    yerr_norm = np.concatenate([meanstd[ptidxs] ])
    y_norm = np.concatenate([mean_p[ptidxs]])
    norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
    # if for too low statistics that no point passes the criteria then plot full y-range
    if np.sum(norm_pos!=0):
        left_lim = np.min((y_norm-yerr_norm)[norm_pos])
        right_lim = np.max((yerr_norm+y_norm)[norm_pos])
        lim_pad = (right_lim - left_lim)/20
        ax.set_ylim(left_lim-lim_pad, right_lim+lim_pad*4)
    ax.set_xlabel(r'$|\eta|$');
    ax.set_ylabel(r'median response');
#     ax.set_xscale('log')
    ax.legend(loc='lower left')
    
    tag = tag[4:]
    FitFigDir1 = 'fig/corr_vs_pt'
    FitFigDir2 = FitFigDir1+'/'+tag
    FitFigDir2 = legend_str_to_filename(FitFigDir2)
    if not os.path.exists(FitFigDir1):
        os.mkdir(FitFigDir1)
    if not os.path.exists(FitFigDir2):
        os.mkdir(FitFigDir2)

    # hep.cms.label("Preliminary", loc=0, data=False, ax=ax)
    hep.cms.label("Private work", loc=0, data=False, ax=ax, rlabel='')
    hep.label.exp_text(text=f'{flavor} jets\n{tag} sample', loc=2, ax=ax)
    # hep.label.exp_text(text=f'{flavor} jets, {tag} sample', loc=0, ax=ax)
    figname = FitFigDir2+'/corr_vs_eta_'+tag+'_'+flavor
    figname = legend_str_to_filename(figname)
    plt.savefig(figname+'.pdf');
    plt.savefig(figname+'.png');
    print(f'Figure saved: {figname}.pdf /.png')
    plt.show();
    
from helpers import gauss
import matplotlib.ticker as ticker
def plot_response_dist(histo, p2, fitlims, figName, dataset_name, hep_txt='', txt2print='', print_txt=True ):
        
    xvals = histo.axes[0].centers
    f_xvals = np.linspace(xvals[fitlims[0]],xvals[fitlims[1]],5001)
    fgaus = gauss(f_xvals, *p2)
    
    f_xvals_full = np.linspace(0,max(xvals),5001)
    fgaus_full = gauss(f_xvals_full, *p2)
    # edd = histo.axes['ptresponse'].edges
    # histo = histo[:len(edd)-2] 
    
    fig, ax2 = plt.subplots();
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    # hep.histplot(histo.values(), histo.axes[0].edges-0.20, yerr=np.sqrt(histo.variances()), label=dataset_name, histtype='fill', alpha=0.6, color=colors[0])
    plot = histo.plot1d(ax=ax2, label=dataset_name, histtype='fill', alpha=0.6, color=colors[0])
    plot = histo.plot1d(ax=ax2, histtype='errorbar', alpha=0.6, color=colors[0], linewidth=1.1, markersize=0)
    # ax2.plot(f_xvals, fgaus, label='Gaussian fit', markersize=0, linewidth=1.8, color=colors[1])
    ax2.plot(f_xvals, fgaus, label='fit', markersize=0, linewidth=1.8, color=colors[1])
    ax2.plot(f_xvals_full, fgaus_full, '--', markersize=0, linewidth=1.2, color=colors[1])
    ax2.set_xlabel("Response ($p_{T,reco}/p_{T,ptcl}$)")
    ax2.set_ylabel("Events")
    max_lim = np.min([np.max(xvals), 2.0])
    ax2.set_xlim([0,max_lim])
    # ax2.set_xlim([0.2,1.8])
    ylim = ax2.get_ylim()
    # Use the scientific notation already from 4 zeros
    formatter = ticker.ScalarFormatter(useMathText=False)
    formatter.set_powerlimits((-4, 4))
    ax2.yaxis.set_major_formatter(formatter)
    yh = (ylim[1]-ylim[0])
    ax2.set_ylim(ylim[0], ylim[1]+yh*0.2 )

    # #### for the poster
    # ax2.vlines(0.996, 1.0e-5, 1.9e-5, linestyles='-',color="black",
    #         linewidth=2.2,)
    # ax2.text(1.10, 1.55e-5, r'median $\approx$ mean '+f'\n'+r'=$0.996\pm0.004$ ', fontsize=11, color='black')

    if print_txt:
        hep_txt+=txt2print
    ax2.legend()
    
    # hep.label.exp_text(text=hep_txt, loc=0)
    hep.cms.label("Private work", loc=0, data=False, ax=ax2, rlabel='')
    hep.label.exp_text(text=hep_txt, loc=2)
    plt.savefig(figName+'.png', dpi=plt.rcParamsDefault['figure.dpi']);
    plt.savefig(figName+'.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
    plt.show(); 


def plot_response_dist_stack(h_stack, p2, figName, dataset_name, hep_txt='', txt2print='', print_txt=True ):
    xvals = h_stack.axes[0].centers
    f_xvals = np.linspace(0,max(xvals),5001)
    fgaus2 = gauss(f_xvals, *p2)
    # edd = histo.axes['ptresponse'].edges
    # histo = histo[:len(edd)-2] 
    
    fig, ax2 = plt.subplots();
    plot = h_stack.plot(stack=True, ax=ax2, histtype='step', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(h_stack)])
    # plot = histo.plot1d(ax=ax2, label='dataset', histtype='fill', alpha=0.6)
    ax2.plot(f_xvals, fgaus2, label='Gaussian fit', markersize=0, linewidth=1.8)
    ax2.set_xlabel("Response ($p_{T,reco}/p_{T,ptcl}$)")
    max_lim = np.min([np.max(xvals), 2.0])
    ax2.set_xlim([0,max_lim])
    ylim = ax2.get_ylim()
    yh = (ylim[1]-ylim[0])
    ax2.set_ylim(ylim[0], ylim[1]+yh*0.2 )
    if print_txt:
        hep_txt+=txt2print
    ax2.legend();
    
    hep.label.exp_text(text=hep_txt, loc=0)
    hep.label.exp_text(text=hep_txt, loc=2)
    plt.savefig(figName+'.png', dpi=plt.rcParamsDefault['figure.dpi']);
    plt.savefig(figName+'.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
    plt.show();   