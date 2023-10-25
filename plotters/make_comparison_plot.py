import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline
from coffea import util
from cycler import cycler
import os
import mplhep as hep
from fileNames.available_datasets import legend_labels
    
from common_binning import JERC_Constants
from JetEtaBins import JetEtaBins, PtBins

def make_comparison_plot(data_dict,
                              function_dict,
                              etabins=JetEtaBins("HCalPart", absolute=True), #np.array(JERC_Constants.etaBinsEdges_CaloTowers_full()),
                              ptbins=PtBins("MC_truth"), #np.array(JERC_Constants.ptBinsEdgesMCTruth()),
                              binidx=0, flav='b',
                              ratio_name='ratio',
                              inverse=True, plotvspt=True, ratio_ylim=None):
    ''' Make a plot of the jet energy response vs pt for the data points in the dictionary `data_dict`
    Compare with with lines obtained from coffea evaluators in `function_dict`.
    `function_dict` can be None than, no lines are shown.
    A ratio plot is drawn vs the first entry in `data_dict`.
    For the legends and figure name the flavor `flav` and eta bin values with the index `etaidx` are used.
    '''
   
    keys = [key for key in data_dict.keys()]
    start = ptbins.get_bin_idx(20) if plotvspt else 0 #np.searchsorted(ptbins, 20, side='left')
    end = ptbins.nbins if plotvspt else etabins.nbins

    data_range = tuple([range(start,end),binidx]) if plotvspt else tuple([binidx, range(start,end)])
    yvals = np.array([key[0][data_range] for key in data_dict.values()])
    # end = np.min([len(yv) for yv in yvals])+start #cut, so that all yvals are the same
    stds  = np.array([key[1][data_range] for key in data_dict.values()])
    reco_pts  = np.array([key[2][data_range] if len(key[2].shape)==2 else key[2][data_range] for key in data_dict.values()])

#     if not plotvspt:
#         yvals = yvals.T
#         stds = stds.T
#         reco_pts = reco_pts.T

    ### Replacing response values to corrections
    use_recopt=inverse
    if not plotvspt:
        use_recopt=False
        
    bins = etabins if plotvspt else ptbins

    # yvals_base[(yvals_base==0) | (np.abs(yvals_base)==np.inf)] = np.nan
    yvals[(yvals==0) | (np.abs(yvals)==np.inf)] = np.nan

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Set up a new cycler to set up colors of the lines match to the colors of the points and all start with black
    # , while for the ratio plot to start with the second marker and color
    reset_colors = True ### if match the colors of the first points to the colors of the first lines
    old_rc_cycler = plt.rcParams['axes.prop_cycle']
    rc_bykey = old_rc_cycler.by_key()
    for key in rc_bykey.keys():
        if key=='color':
            rc_bykey[key] = ['k']+rc_bykey[key]
        else:
            rc_bykey[key] = rc_bykey[key]+[rc_bykey[key][-1]]
    new_cycler = cycler(**rc_bykey)
    ax.set_prop_cycle(new_cycler)
    
    for axis in [ax.xaxis, ax.yaxis, ax2.xaxis, ax2.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    if plotvspt:
        xvals = reco_pts if use_recopt else np.array([ptbins.centres[start:end]]*len(yvals))
    else:
        xvals = np.array([etabins.centres[start:end]]*len(yvals))
    validx = (xvals>0)*(yvals>0)
    # xvals_cont = {name: np.geomspace(np.min(xv[valx]), np.max(xv[valx]), 100)
    #               for xv, valx, name in zip(xvals, validx, function_dict.keys())}
    
    linspacefun = np.geomspace if plotvspt else np.linspace
    if np.sum(validx) != 0:
        xvals_cont = linspacefun(np.min(xvals[validx]), np.max(xvals[validx]), 100)
    else:
        xvals_cont = linspacefun(np.min(xvals), np.max(xvals), 100)
                                    # for name in function_dict.keys()}
    ### values for splines can only be in a valid range while for corrections evaluators they can go out of range
    ### so one needs to define two xvals
    validx_all = np.logical_not(np.any(np.logical_not(validx), axis=0))
    if np.sum(validx_all) == 0:
        validx_all = np.ones(validx_all.shape)==1
    xspline = linspacefun(np.min(xvals[0,validx_all]),  np.max(xvals[0,validx_all]), 100)
    xlog10_spline = np.log10(xspline)

    bins2 = ptbins if plotvspt else etabins
    wd = np.abs(np.diff(bins2.edges))[start:end] #bin_widths

    if inverse==True:
        yvals = 1/yvals
        ### Error propagation
        stds = yvals**2*stds

    if plotvspt:
        eta_str = r'{:0.2f}$<|\eta|<${:0.2f}'.format(etabins.edges[binidx], etabins.edges[binidx+1])
    else:
        eta_str = 'to be implemented'
    p1 = ax.errorbar(xvals[0], yvals[0], yerr=stds[0], #marker='o',
                     capsize=1.6, capthick=0.7, linewidth=1.0,
    #                      markerfacecolor='none', markeredgewidth=1,
                     linestyle="none", label=keys[0]) #+', '+eta_str)

    # markers = ['v','^','d', 'p']
    for xval, yval, name, std in zip(xvals[1:], yvals[1:], keys[1:], stds[1:]):
        ax.errorbar(xval, yval, yerr=std, #marker=marker,
                    capsize=1.6, capthick=0.7, linewidth=1.0,
                    linestyle="none", label=name)

    if reset_colors:
        ax.set_prop_cycle(new_cycler)

#     assert False
    yvals_cont = {}
    yvals_spline = {}
    for name in function_dict.keys():
        correction_fnc, closure = function_dict[name]
        xv_cont = xvals_cont
        if closure is None or closure==1:
            def closure(a,b):
                return np.ones_like(a*b)
        #to ensure that the correction is applied from the right side of the bin border
        binval = bins.edges[binidx]+0.001
        vals_cont = (np.array([binval]), xv_cont) if plotvspt else (xv_cont, np.array([binval]))
        vals_spline = (np.array([binval]), xspline) if plotvspt else (xspline, np.array([binval]))
        yvals_cont[name] = correction_fnc(*vals_cont)/closure(*vals_cont)
        yvals_spline[name] = correction_fnc(*vals_spline)/closure(*vals_spline)
        
        eta_str = ''
        if plotvspt:
            corr_etabins = correction_fnc._bins['JetEta'] 
            corr_bin_idx = np.searchsorted(corr_etabins, binval, side='right')-1
    #         assert False
            if corr_bin_idx==len(corr_etabins):
                corr_bin_idx-=1
            if 'Winter' in name:
                eta_str = ', \n'+r' {:0.1f}$<|\eta|<${:0.1f}'.format(corr_etabins[corr_bin_idx], corr_etabins[corr_bin_idx+1])
            
        ax.plot(xv_cont, yvals_cont[name], label=name+eta_str, markersize=0) # +', '+eta_str, markersize=0)

    if not plotvspt:
        for HCal_border in JERC_Constants.etaBinsEdges_Win14():
            ax.vlines(HCal_border,0, 2, linestyles='--',color="gray",
                linewidth=1,)

    # assert False

    ############################ Data ratio plot ######################################
    
    ax2.hlines(1,-10, 10000, linestyles='--',color="black", 
               linewidth=1,)
    if not plotvspt:
        for HCal_border in JERC_Constants.etaBinsEdges_Win14():
            ax2.vlines(HCal_border,0, 2, linestyles='--',color="gray",
                linewidth=1,)


#     else:
#         ax2.hlines(1,-10, 10, linestyles='--',color="black",
#             linewidth=1,)

    
    data_model_ratio = yvals/yvals[0]
    data_model_ratio_unc = stds / yvals[0]
    
    non_nan_ratio = ~np.isnan(data_model_ratio_unc[0])

    ax2.bar(
        xvals[0, non_nan_ratio],
        2 * data_model_ratio_unc[0][non_nan_ratio],
        width=wd[non_nan_ratio],
        bottom=1.0 - data_model_ratio_unc[0][non_nan_ratio],
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=10 * "/",
    )

    rc_bykey = new_cycler.by_key().copy()
    for key in rc_bykey:
        rc_bykey[key] = rc_bykey[key][1:]
    shifted_cycler = cycler(**rc_bykey)
    ax2.set_prop_cycle(shifted_cycler)

    for xval, val, std in zip(xvals[1:], data_model_ratio[1:], data_model_ratio_unc[1:]): #, markers):
        ax2.errorbar(
            xval,
            val, #[nonzero_model_yield],
            yerr=std, #[nonzero_model_yield],
            linestyle="none",
            capsize=1.6, capthick=0.7, linewidth=1.0,
            #fmt=marker,
        )

    ############################ Curves in the ratio plot (using spline approximation) #######
    if reset_colors:
        ax2.set_prop_cycle(new_cycler)
        
    if np.sum(validx) != 0:
        spline_func = CubicSpline(np.log10(xvals[0][validx[0]]), yvals[0][validx[0]], bc_type='natural')
    else:
        spline_func = CubicSpline([1,1], [-5,5], bc_type='natural') if plotvspt else CubicSpline([1,1], [20,100], bc_type='natural')
    y_spline = spline_func(xlog10_spline)

    for key in yvals_spline.keys():
        ax2.plot(xspline, yvals_spline[key]/y_spline, markersize=0)

    ax2.set_ylabel(ratio_name)
    ax2.tick_params(axis="both", which="major", pad=8)
    ax2.tick_params(direction="in", top=True, right=True, which="both")
    fig.set_tight_layout(True)

    ######################## Calculate resonable limits excluding the few points with insane errors ############################
    if np.sum(validx) != 0:
        ### Recalculate the limits for the top ax
        yerr_norm = np.concatenate([stds])
        y_norm = np.concatenate([yvals])
        norm_pos = (yerr_norm<0.04) &  (yerr_norm != np.inf) & (y_norm>-0.1)
        left_lim = np.min((y_norm-yerr_norm)[norm_pos])
        right_lim = np.max((yerr_norm+y_norm)[norm_pos])
        lim_pad = (right_lim - left_lim)/2.5
        ax.set_ylim(left_lim-lim_pad/10, right_lim+lim_pad)

        ### Recalculate the limits for the ratio plot
        yerr_norm = np.concatenate(data_model_ratio_unc)
        y_norm = np.concatenate(data_model_ratio)
        norm_pos = (yerr_norm<0.4) &  (yerr_norm != np.inf) & (y_norm>-0.1)
        if ~np.any(norm_pos):
            print("Cannot determine ylimits")
            # norm_pos = np.ones(len(yerr_norm), dtype=int)
        else:
            # raise Exception("Cannot determine ylimits")
            left_lim = np.min((y_norm-yerr_norm)[norm_pos])
            right_lim = np.max((yerr_norm+y_norm)[norm_pos])
            lim_pad = (right_lim - left_lim)/5
            ax2.set_ylim(left_lim-lim_pad, right_lim+lim_pad)
            # print(f"normal pos = {norm_pos}")
            # print(f"right lim = {right_lim}")

    if not ratio_ylim==None:
        ax2.set_ylim(ratio_ylim)

    if plotvspt:
        xlabel = r'$p_{T,reco}$ (GeV)' if use_recopt else r'$p_{T,ptcl}$ (GeV)'
    else:
        xlabel = r'$|\eta|$'
    ax2.set_xlabel(xlabel);
    ylabel = r'correction (1/median)' if inverse else r'median response'
    ax.set_ylabel(ylabel);
    if plotvspt:
        ax.set_xscale('log')
        ax2.set_xscale('log')

    xlims = ax.get_xlim()
    # assert False
    ax.hlines(1,-10, 10000, linestyles='--',color="black",
              linewidth=1,)

#     ax.set_xticks([])
    if plotvspt:
        ax2.set_xticks([10, 20, 50, 100, 500, 1000, 5000]) 
    ax.set_xticks(ax2.get_xticks())
    ax.set_xticklabels([])
    ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    leg1 = ax.legend(loc="upper right", ncol=1)

    if not plotvspt:
        xlims = (-0.2, 5.3)
    ax.set_xlim(xlims)
    ax2.set_xlim(xlims)

    eta_string = bins.idx2str(binidx) #'_eta'+str(etabins_abs[etaidx])+'to'+str(etabins_abs[etaidx+1])
#     eta_string = eta_string.replace('.','')
    fig_corr_name = 'corr' if inverse else 'med_resp'
    fig_x_str = 'pt' if plotvspt else 'eta'
    run_name =  f'{fig_corr_name}_vs_{fig_x_str}_L5_'+'-'.join(keys)+'-'.join(function_dict.keys())
    run_name = (run_name.replace(legend_labels["ttbar"]["lab"], 'ttbar').replace(', ', '-')
                .replace(" ", "_").replace("+", "_").replace('(', '').replace(')', '').replace('/', '').replace('\n', '').replace('$', '').replace('\\', '')
    )
    dir_name1 = f'fig/{fig_corr_name}_vs_{fig_x_str}_comparisons/'
    dir_name2 = dir_name1+run_name
    if not os.path.exists(dir_name1):
        os.mkdir(dir_name1)
        print("Creating directory ", dir_name1)
    if not os.path.exists(dir_name2):
        os.mkdir(dir_name2)
        print("Creating directory ", dir_name2)

    hep.cms.label("Private work", loc=0, data=False, ax=ax, rlabel='')
    # hep.cms.label("Preliminary", loc=0, data=False, ax=ax)
    hep.label.exp_text(text=f'{bins.idx2plot_str(binidx)}\n{flav} jets', loc=2, ax=ax)
    fig_name = dir_name2+'/'+run_name+"_"+flav+'_'+eta_string
    print("Saving plot for eta = ", eta_string)
    print("Saving plot with the name = ", fig_name+".pdf / .png")
    plt.savefig(fig_name+'.pdf')
    plt.savefig(fig_name+'.png')
    plt.show()


from helpers import read_data as read_data_orig
def read_data(mean_name, flav, tag1):
    return read_data_orig(mean_name, flav, tag1, '../')

### Some recent file to get out the binning
outname = '../out/CoffeaJERCOutputs_L5_QCD-Py.coffea'
output = util.load(outname)
ptbins = output[list(output.keys())[0]]['ptresponse_b'].axes['pt_gen'].edges
ptbins_c = output[list(output.keys())[0]]['ptresponse_b'].axes['pt_gen'].centers
etabins = np.array(JERC_Constants.etaBinsEdges_CaloTowers_full())
# etabins = np.array(JERC_Constants.etaBinsEdges_Aut18_full())
etabins = np.array(JERC_Constants.etaBinsEdges_Win14_full())
    
etabins_abs = etabins[(len(etabins)-1)//2:]
etabins_c = (etabins_abs[:-1]+etabins_abs[1:])/2 #output['ptresponse'].axis('jeteta').centers()


def make_double_ratio_plot(outputname1, outputname2, etaidx=0, flav='',
                            ratio_name='ratio'):
    ''' Make a double ratio plot for comparing flavor vs anti-flavor responses
    To do:
    make it work with input parameters of ptbins and etabins, not the ones real at the top
    '''
        
    median_1 = read_data("Median", flav, outputname1)
    medianstd_1 = read_data("MedianStd", flav, outputname1)
    median_2 = read_data("Median", flav+'bar', outputname1)
    medianstd_2 = read_data("MedianStd", flav+'bar', outputname1)
    median_3 = read_data("Median", flav, outputname2)
    medianstd_3 = read_data("MedianStd", flav, outputname2)
    median_4 = read_data("Median", flav+'bar', outputname2)
    medianstd_4 = read_data("MedianStd", flav+'bar', outputname2)

        
    yvals_base = median_1
    std_base = medianstd_1

    yvals_ref = median_2
    std_ref = medianstd_2

    yvals_base2 = median_3
    std_base2 = medianstd_3

    yvals_ref2 = median_4
    std_ref2 = medianstd_4


    mean_p_base = yvals_base.copy()
    mean_p_base[(mean_p_base==0) | (np.abs(mean_p_base)==np.inf)] = np.nan

    # mean_ps = []
    # for yvar in yvars:
    #     mean_ps = yvar.copy()

    # yvars[(yvars==0) | (np.abs(yvars)==np.inf)] = np.nan


    # fig = plt.figure()
    # gs = fig.add_gridspec(nrows=1, ncols=1)
    fig, ax2 = plt.subplots();
    # ax2 = fig.add_subplot(gs[1])
    start = np.where(ptbins<=20)[0][-1]

    for axis in [ax2.xaxis, ax2.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    xvals = (ptbins[start:-1] + ptbins[start+1:])/2
    wd = np.abs(ptbins[start:-1] - ptbins[start+1:])

    yvals_base = mean_p_base[start:,etaidx]
    yvals_base[(yvals_base==0) | (np.abs(yvals_base)==np.inf)] = np.nan
    std_base = std_base[start:,etaidx]

    yvals_ref = yvals_ref[start:,etaidx]
    yvals_ref[(yvals_ref==0) | (np.abs(yvals_ref)==np.inf)] = np.nan
    std_ref = std_ref[start:,etaidx]

    yvals_base2 = yvals_base2[start:,etaidx]
    yvals_base2[(yvals_base2==0) | (np.abs(yvals_base2)==np.inf)] = np.nan
    std_base2 = std_base2[start:,etaidx]


    yvals_ref2 = yvals_ref2[start:,etaidx]
    yvals_ref2[(yvals_ref2==0) | (np.abs(yvals_ref2)==np.inf)] = np.nan
    std_ref2 = std_ref2[start:,etaidx]


    # markers = ['v','^','d', 'p']
    # for val, name, std, marker in zip(yvars, names, stds, markers):
    #     ax.errorbar(xvals, val, yerr=std, marker=marker,
    #                 linestyle="none", label=name)

    rel_mc_unc =  std_base/yvals_base 

    ax2.bar(
        xvals,
        2 * rel_mc_unc,
        width=wd,
        bottom=1.0 - rel_mc_unc,
        fill=False,
        linewidth=0,
        alpha=0.9,
        edgecolor="red",
        hatch=10 * "/",
    )

    rel_mc_unc2 =  std_base2/yvals_base2 

    ax2.bar(
        xvals+0.00001,
        2 * rel_mc_unc2,
        width=wd,
        bottom=1.0 - rel_mc_unc2,
        fill=False,
        linewidth=0,
        alpha=0.5,
        edgecolor="blue",
        hatch=10 * "\\",
    )


    # data in ratio plot
    data_model_ratio = yvals_ref/yvals_base
    data_model_ratio_unc = std_ref / yvals_base

    # for val, std, marker in zip(data_model_ratio, data_model_ratio_unc, markers):
    ax2.errorbar(
        xvals,
        data_model_ratio, #[nonzero_model_yield],
        yerr=data_model_ratio_unc, #[nonzero_model_yield],
        capsize=1.6, capthick=0.7, linewidth=1.0,
        fmt='o',
        label = 'Pythia',
    )

    # data in ratio plot
    data_model_ratio2 = yvals_ref2/yvals_base2
    data_model_ratio_unc2 = std_ref2/ yvals_base2

    ax2.errorbar(
        xvals,
        data_model_ratio2, #[nonzero_model_yield],
        yerr=data_model_ratio_unc2, #[nonzero_model_yield],
        capsize=1.6, capthick=0.7, linewidth=1.0,
        fmt='^',
        label = 'Herwig',
    #     colour='blue'
    )

    ax2.set_ylabel(ratio_name)
    ax2.tick_params(axis="both", which="major", pad=8)
    ax2.tick_params(direction="in", top=True, right=True, which="both")

    fig.set_tight_layout(True)



    # ### Calculate resonable limits excluding the few points with insane errors
    # yerr_norm = np.concatenate([std_base, std_base2 ])
    # y_norm = np.concatenate([vals_base, vals_base2])
    # norm_pos = (yerr_norm<0.04) &  (yerr_norm != np.inf) & (y_norm>-0.1)
    # ax.set_ylim(np.min((y_norm-yerr_norm)[norm_pos]), np.max((yerr_norm+y_norm)[norm_pos]))

    yerr_norm = np.concatenate([rel_mc_unc, rel_mc_unc2, data_model_ratio_unc, data_model_ratio_unc2 ])
    y_norm = np.concatenate([yvals_base/yvals_base, yvals_base2/yvals_base2, data_model_ratio, data_model_ratio2])
    norm_pos = (yerr_norm<0.003) &  (yerr_norm != np.inf) & (y_norm>-0.1)  
    #     if flav == '_b' and k==3:
    #         1/0
    left_lim = np.min((y_norm-yerr_norm)[norm_pos])
    right_lim = np.max((yerr_norm+y_norm)[norm_pos])
    lim_pad = (right_lim - left_lim)/10
    ax2.set_ylim(left_lim-lim_pad, right_lim+lim_pad)

    ax2.set_xlabel(r'$p_T$ (GeV)');
    # ax.set_ylabel(r'median response');
    # ax.set_xscale('log')
    ax2.set_xscale('log')

    # ax.set_xticks([])

    good_xlims = ax2.get_xlim()

    ax2.hlines(1,1, 10000, linestyles='--',color="black",
        linewidth=1,)
    ax2.set_xticks([20, 50, 100, 500, 1000, 5000])
    ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax2.set_xlim(good_xlims)

    leg1 = ax2.legend()


    eta_string = '_eta'+str(etabins_abs[etaidx])+'to'+str(etabins_abs[etaidx+1])
    eta_string = eta_string.replace('.','')
    print("Saving plot for eta = ", eta_string)
    fig_name = 'fig/corr_vs_pt'+flav+eta_string+'_L5_double_ratio'+'-median'
    fig_name = fig_name.replace('$t\overline{\, t}$', 'ttbar').replace(', ', '-').replace('(', '').replace(')', '')
    print("Saving plot with the name = ", fig_name)
    plt.savefig(fig_name+'.pdf');
    plt.savefig(fig_name+'.png');
    # gs1.tight_layout(fig, rect=[0, 0.1, 0.8, 0.5])
    plt.show();

