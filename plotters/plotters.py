import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_corrections(mean, meanstd, ptbins_c, etabins, savename):

    ### To ignore the points with 0 on y axis when setting the y axis limits
    mean_p = mean.copy()
    mean_p[mean_p==0] = np.nan

    fig, ax = plt.subplots()
    start = np.searchsorted(ptbins_c, 20, side='left') #np.where(ptbins<=20)[0][-1]
    
    #### four different eta bins to plot
    k2 = np.where(etabins<=0)[0][-1]
    k4 = np.where(etabins<=1.3)[0][-1]
    k6 = np.where(etabins<=2.5)[0][-1]
    k8 = np.where(etabins<=3.0)[0][-1]
#     lastbin = np.where(~ np.isnan(mean_p[:, k2]*mean_p[:, k4]*mean_p[:, k6]*mean_p[:, k8]))[0]
#     lastbin = lastbin[-1] if len(lastbin)>0 else -1
    lastbin = -1
    
    ptbins_plot = ptbins_c[start:lastbin]
    meanstd = meanstd[start:lastbin,:]
    
    mean_p = mean_p[start:lastbin]
    
#     print(f'ptvinsplot = {ptbins_plot}')
    plt.errorbar(ptbins_plot, mean_p[:,k2], yerr=meanstd[:,k2], marker='o',
                 linestyle="none", label=f'{etabins[k2]}'+r'$<\eta<$'+f'{etabins[k2+1]}')
    plt.errorbar(ptbins_plot, mean_p[:,k4], yerr=meanstd[:,k4], marker='o',
             linestyle="none", label=f'{etabins[k4]}'+r'$<\eta<$'+f'{etabins[k4+1]}')
    plt.errorbar(ptbins_plot, mean_p[:,k6], yerr=meanstd[:,k6], marker='o',
             linestyle="none", label=f'{etabins[k6]}'+r'$<\eta<$'+f'{etabins[k6+1]}')
    plt.errorbar(ptbins_plot, mean_p[:,k8], yerr=meanstd[:,k8], marker='o',
             linestyle="none", label=f'{etabins[k8]}'+r'$<\eta<$'+f'{etabins[k8+1]}')

    ### Calculate resonable limits excluding the few points with insane errors
    yerr_norm = np.concatenate([meanstd[:,[k2, k4, k6, k8]] ])
    y_norm = np.concatenate([mean_p[:,[k2, k4, k6, k8]]])
    norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
    
    left_lim = np.min((y_norm-yerr_norm)[norm_pos])
    right_lim = np.max((yerr_norm+y_norm)[norm_pos])
    lim_pad = (right_lim - left_lim)/20
    ax.set_ylim(left_lim-lim_pad, right_lim+lim_pad)
#     ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]) ,np.max((yerr_norm+y_norm)[norm_pos]))
    
    ax.set_xscale('log')
    
    good_xlims = ax.get_xlim()
    ax.set_xticks([20, 50, 100, 500, 1000, 5000])
    ax.set_xlim(good_xlims)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel(r'$p_T$ (GeV)');
    ax.set_ylabel(r'median response');
    ax.legend()
    figname = 'fig/corr_vs_pt_'+savename
    plt.savefig(figname+'.pdf');
    plt.savefig(figname+'.png');
    print(f'Figure saved: {figname}.pdf /.png')

    plt.show();


def plot_corrections_eta(mean, meanstd, ptbins, etabins_c, savename):
    ### To ignore the points with 0 on y axis when setting the y axis limits
    mean_p = mean.copy()
    mean_p[mean_p==0] = np.nan

    fig, ax = plt.subplots()
#     start = np.where(ptbins<=20)[0][-1]
    
#     ptbins_plot = ptbins_c[start:]
#     meanstd = meanstd[start:,:]
#     mean_p = mean_p[start:]
    
    k2 = np.where(ptbins<=15)[0][-1]
    k4 = np.where(ptbins<=40)[0][-1]
    k6 = np.where(ptbins<=150)[0][-1]
    k8 = np.where(ptbins<=400)[0][-1]
    
#     np.isnan(mean_p[k2,:]*mean_p[k4,:]*mean_p[k6,:]*mean_p[k8,:])
    
    plt.errorbar(etabins_c, mean_p[k2,:], yerr=meanstd[k2], marker='o',
                 linestyle="none", label=f'{ptbins[k2]}'+r'$<p_t<$'+f'{ptbins[k2+1]}')
    plt.errorbar(etabins_c, mean_p[k4,:], yerr=meanstd[k4], marker='o',
             linestyle="none", label=f'{ptbins[k4]}'+r'$<p_t<$'+f'{ptbins[k4+1]}')
    plt.errorbar(etabins_c, mean_p[k6], yerr=meanstd[k6], marker='o',
             linestyle="none", label=f'{ptbins[k6]}'+r'$<p_t<$'+f'{ptbins[k6+1]}')
    plt.errorbar(etabins_c, mean_p[k8], yerr=meanstd[k8], marker='o',
             linestyle="none", label=f'{ptbins[k8]}'+r'$<p_t<$'+f'{ptbins[k8+1]}')

    ### Calculate resonable limits excluding the few points with insane errors
    yerr_norm = np.concatenate([np.sqrtmeanstd[[k2, k4, k6, k8]] ])
    y_norm = np.concatenate([mean_p[[k2, k4, k6, k8]]])
    norm_pos = (yerr_norm<0.02) &  (yerr_norm != np.inf) & (y_norm>-0.1)
    ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]) ,np.max((yerr_norm+y_norm)[norm_pos]))
    ax.set_xlabel(r'$\eta$');
    ax.set_ylabel(r'median response');
#     ax.set_xscale('log')
    ax.legend()
    
    figname = 'fig/corr_vs_pt_'+savename
    plt.savefig(figname+'.pdf');
    plt.savefig(figname+'.png');
    print(f'Figure saved: {figname}.pdf /.png')
    plt.show();
    
from helpers import gauss
def plot_response_dist(histo, xvals, p2, cov, chi2, Ndof, median, medianstd, Neff, figName ):
    width_ik = np.abs(p2[2])
    f_xvals = np.linspace(0,max(xvals),5001)
    fgaus2 = gauss(f_xvals, *p2)
    edd = histo.axes['ptresponse'].edges
    histo = histo[1:len(edd)-2] 
    #remove the underflow, overflow. Not sure if in hist.hist it is stored in the last and first bin like in coffea.hist
#     histo = histo[{'ptresponse', blah}]
    
    fig, ax2 = plt.subplots();
#     hist.plot1d(histo, ax=ax2, overlay='dataset', overflow='all',
#                 fill_opts={'alpha': .5, 'edgecolor': (0,0,0,0.3), 'linewidth': 1.4})
    plot = histo.plot1d(ax=ax2, label='dataset', histtype='fill', alpha=0.6)
#     plot[0].set_linewidth(2)
    ax2.plot(f_xvals, fgaus2, label='Gaus',linewidth=1.8)
    ax2.set_xlabel("Response ($p_{T,reco}/p_{T,ptcl}$)")
    max_lim = np.searchsorted(xvals, 2.0)
    ax2.set_xlim([0,max_lim])
    h = ax2.get_ylim()[1]/1.05
    plt.text(0.03,0.95*h,r'Mean {0:0.3f}$\pm${1:0.3f}'.format(p2[1], np.sqrt(cov[1,1])))
    plt.text(0.03,0.88*h,r'Width {0:0.3f}$\pm${1:0.3f}'.format(width_ik, np.sqrt(cov[2,2])))
    plt.text(0.03,0.81*h,r'Median {0:0.3f}$\pm${1:0.3f}'.format(median, medianstd))
    plt.text(0.03,0.73*h,r'$\chi^2/ndof$ {0:0.2g}/{1:0.0f}'.format(chi2, Ndof))
    plt.text(0.03,0.66*h,r'Neff = {0:0.3g}'.format(Neff))
    ax2.legend();

    
    plt.savefig(figName+'.png', dpi=plt.rcParamsDefault['figure.dpi']);
    plt.savefig(figName+'.pdf', dpi=plt.rcParamsDefault['figure.dpi']);
#     print("Saving to: ", figName+'.png')
#     plt.close();   
    