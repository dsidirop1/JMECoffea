import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline
from coffea import util
import cycler
    

### Some recent file to get out the binning
outname = 'out/CoffeaJERCOutputs_L5_fine_etaBins_QCD.coffea'
output = util.load(outname)

f_xvals = np.linspace(0,5,5001)

ptbins = output['ptresponse'].axis('pt').edges()
ptbins_c = output['ptresponse'].axis('pt').centers()
etabins = np.array([-5, -3, -2.5, -1.3, 0, 1.3, 2.5, 3, 5])
# etabins = np.array([-5.191, -3.489, -3.139, -2.853,   -2.5,
#                     -2.322,  -1.93, -1.653, -1.305, -0.783,      
#                     0,  0.783,  1.305,  1.653,   1.93,  2.322, 2.5, 
#                     2.853,  3.139,  3.489, 5.191])
    

# jetpt_length = len(ptbins)-1
# jeteta_length = (len(etabins)-1)//2

etabins_mod = etabins[(len(etabins)-1)//2:]
etabins_c = (etabins_mod[:-1]+etabins_mod[1:])/2 #output['ptresponse'].axis('jeteta').centers()

def make_comparison_plot(data_dict,
                              function_dict,
                              etaidx=0, samp='',
                              ratio_name='ratio'):
    ''' Make a coparison and a ratio plot of yvar2 vs yvar_base
    
    To do:
    - ptreco for all samples
    - ratio at least for data
    - ratio also for fuctions at each point
    - Proper errors on closure data stuff
    '''
   
    keys = [key for key in data_dict.keys()]
    start = np.searchsorted(ptbins, 15, side='left')

    yvals = np.array([key[0][start:,etaidx] for key in data_dict.values()])
    stds  = np.array([key[1][start:,etaidx] for key in data_dict.values()])
    reco_pts  = np.array([key[2][start:,etaidx] if len(key[2].shape)==2 else key[2][start:] for key in data_dict.values()])


    ### Replacing response values to corrections
    inverse=False
    use_recopt=False

    # yvals_base[(yvals_base==0) | (np.abs(yvals_base)==np.inf)] = np.nan
    yvals[(yvals==0) | (np.abs(yvals)==np.inf)] = np.nan

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # To make colors colors of the first points to the colors of the first lines and all start with black
    reset_colors = True ### if match the colors of the first points to the colors of the first lines
    old_rc_cols = plt.rcParams['axes.prop_cycle']
    new_cols = cycler.concat(cycler.cycler(color=['k']), old_rc_cols)
#     plt.rcParams['axes.prop_cycle'] = 
    ax.set_prop_cycle(new_cols)
    
    for axis in [ax.xaxis, ax.yaxis, ax2.xaxis, ax2.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    xvals = reco_pts[0] if use_recopt else (ptbins[start:-1] + ptbins[start+1:])/2
    xvals_cont = 10**(np.linspace(np.log10(xvals[2]),  np.log10(xvals[np.nonzero(xvals>0)[0][-1]]), 100))
    wd = np.abs(ptbins[start:-1] - ptbins[start+1:]) #bin_widths

    if inverse==True:
        yvals = 1/yvals
        ### Error propagation
        stds = yvals**2*stds

    eta_str = r'{:0.2f}$<\eta<${:0.2f}'.format(etabins_mod[etaidx], etabins_mod[etaidx+1])
    p1 = ax.errorbar(xvals, yvals[0], yerr=stds[0], marker='o',
    #                      markerfacecolor='none', markeredgewidth=1,
                 linestyle="none", label=keys[0]) #+', '+eta_str)

    markers = ['v','^','d', 'p']
    for val, name, std, marker in zip(yvals[1:], keys[1:], stds[1:], markers):
        ax.errorbar(xvals, val, yerr=std, marker=marker,
                    linestyle="none", label=name)
    validx = (xvals>0)*(yvals[0]>0)
    xlog10_spline = np.linspace(np.log10(np.min(xvals[validx])),  np.log10(np.max(xvals[validx])), 100)
    xspline = 10**xlog10_spline

    if reset_colors:
        ax.set_prop_cycle(new_cols)

    yvals_cont = {}
    yvals_spline = {}
    for name in function_dict.keys():
        correction_fnc, closure = function_dict[name]
        if closure is None or closure==1:
            def closure(a,b):
                return np.ones_like(a*b)
        etaval = etabins_mod[etaidx]+0.001 #to insure that the correction is applied from the right side of the bin border
        yvals_cont[name] = correction_fnc(np.array([etaval]), xvals_cont)/closure(np.array([etaval]), xvals_cont)
        yvals_spline[name] = correction_fnc(np.array([etaval]), xspline)/closure(np.array([etaval]), xspline)
        corr_etabins = correction_fnc._bins['JetEta']
        corr_bin_idx = np.searchsorted(corr_etabins, etaval, side='right')-1
#         assert False
        if corr_bin_idx==len(corr_etabins):
            corr_bin_idx-=1
        eta_str = r'{:0.2f}$<\eta<${:0.2f}'.format(corr_etabins[corr_bin_idx], corr_etabins[corr_bin_idx+1])
        ax.plot(xvals_cont, yvals_cont[name], label=name+', '+eta_str)

    ############################ Data ratio plot ######################################
    ax2.hlines(1,1, 10000, linestyles='--',color="black",
        linewidth=1,)
    
    data_model_ratio = yvals/yvals[0]
    data_model_ratio_unc = stds / yvals[0]
    
    non_nan_ratio = ~np.isnan(data_model_ratio_unc[0])

    ax2.bar(
        xvals[non_nan_ratio],
        2 * data_model_ratio_unc[0][non_nan_ratio],
        width=wd[non_nan_ratio],
        bottom=1.0 - data_model_ratio_unc[0][non_nan_ratio],
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=10 * "/",
    )

    for val, std, marker in zip(data_model_ratio[1:], data_model_ratio_unc[1:], markers):
        ax2.errorbar(
            xvals,
            val, #[nonzero_model_yield],
            yerr=std, #[nonzero_model_yield],
            fmt=marker,
        )

    ############################ Curves in the ratio plot (using spline approximation) #######
    if reset_colors:
        ax2.set_prop_cycle(new_cols)
        
    spline_func = CubicSpline(np.log10(xvals[validx]), yvals[0][validx], bc_type='natural')
    y_spline = spline_func(xlog10_spline)

    for key in yvals_spline.keys():
        ax2.plot(xvals_cont, yvals_spline[key]/y_spline)

#     ax.plot(xspline, y_spline)

    ax2.set_ylabel(ratio_name)
    ax2.tick_params(axis="both", which="major", pad=8)
    ax2.tick_params(direction="in", top=True, right=True, which="both")

    fig.set_tight_layout(True)

    ######################## Calculate resonable limits excluding the few points with insane errors
    recalculate_limits=True
    if recalculate_limits:
        yerr_norm = np.concatenate([stds])
        y_norm = np.concatenate([yvals])
        norm_pos = (yerr_norm<0.04) &  (yerr_norm != np.inf) & (y_norm>-0.1)
        ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]), np.max((yerr_norm+y_norm)[norm_pos]) +0.005)

        yerr_norm = np.concatenate(data_model_ratio_unc)
        y_norm = np.concatenate(data_model_ratio)
        norm_pos = (yerr_norm<0.008) &  (yerr_norm != np.inf) & (y_norm>-0.1)  
        if ~np.any(norm_pos):
            print("Cannot determine ylimits")
            norm_pos = np.ones(len(yerr_norm), dtype=int)
            raise Exception("Cannot determine ylimits")
        left_lim = np.min((y_norm-yerr_norm)[norm_pos])
        right_lim = np.max((yerr_norm+y_norm)[norm_pos])
        lim_pad = (right_lim - left_lim)/5
        ax2.set_ylim(left_lim-lim_pad, right_lim+lim_pad)

        
    xlabel = r'$p_{T,reco}$ (GeV)' if use_recopt else r'$p_{T,ptcl}$ (GeV)'
    ax2.set_xlabel(xlabel);
    ylabel = r'correction (1/median)' if inverse else r'median response'
    ax.set_ylabel(ylabel);
    ax.set_xscale('log')
    ax2.set_xscale('log')

    xlims = ax.get_xlim()
    ax.hlines(1,1, 10000, linestyles='--',color="black",
              linewidth=1,)

    ax.set_xticks([])
    ax2.set_xticks([10, 20, 50, 100, 500, 1000, 5000])
    ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    leg1 = ax.legend(ncol=1)

    ax.set_xlim(xlims)
    ax2.set_xlim(xlims)

    eta_string = '_eta'+str(etabins_mod[etaidx])+'to'+str(etabins_mod[etaidx+1])
    eta_string = eta_string.replace('.','')
    print("Saving plot for eta = ", eta_string)
    fig_name = 'fig/corr_vs_pt_'+samp+eta_string+'_L5_'+keys[0]+'-'+'-'.join(function_dict.keys())+'-median'
    fig_name = fig_name.replace(', ', '-').replace(" ", "_")
    print("Saving plot with the name = ", fig_name)
    plt.savefig(fig_name+'.pdf');
    plt.savefig(fig_name+'.png');
    plt.show();