import matplotlib.pyplot as plt

import matplotlib as mpl
import os
import mplhep as hep
import numpy as np
from uncertainty_helpers import get_ratio, ptmin_global, ptmax_global
from coffea.lookup_tools import extractor
from scipy.optimize import curve_fit

color_scheme = {key: cycler_vals
    for cycler_vals, key in zip(plt.rcParams['axes.prop_cycle'], ['g', 'ud', 'c', 'b', 'QCD', 'DY', 'TTBAR', 'DY200', 'unmatched', 's', 'q'])}
color_scheme_antiflav = {key: cycler_vals
    for cycler_vals, key in zip(plt.rcParams['axes.prop_cycle'], ['g', 'udbar', 'cbar', 'bbar', 'QCD', 'DY', 'TTBAR', 'DY200', 'unmatched', 'sbar', 'qbar'])}
color_scheme = color_scheme | color_scheme_antiflav

legend_dict = {'g': 'Gluons', 'q': 'Quarks', 'ud':'UpDown', 'b': 'Bottom', 'c': 'Charm', 's': 'Strange', 'unmatched': 'Unmatched'}
legend_dict_short = {'g': 'g',
                     'ud': 'ud', 'q':'q', 'b': 'b', 'c': 'c', 's':'s',
                     'unmatched': 'unmatched',
                     'udbar': '$\overline{ud}$', 'qbar':'$\overline{q}$', 'bbar': '$\overline{b}$', 'cbar': '$\overline{c}$', 'sbar':'$\overline{s}$'}

def plot_Efractions(sampledict, etaidx, jeteta_bins, ptbins, saveplot=False):
    samples = list(sampledict.keys())
    ptbins_c = ptbins.centres

#     ### Check that Herwig is the first sample and Pythia the second
#     if not ('Her' in samples[0] and 'Py' in samples[1]):
#         raise ValueError('key in the dictionary happened to get reversed')
    
    
    qfracs0, qfrac_var0, spline0, spline2D0 = sampledict[samples[0]]
    qfracs1, qfrac_var1, spline1, spline2D1 = sampledict[samples[1]]
    
    plot_range = range(0, np.searchsorted(ptbins_c,1250)) if 'DY' in "".join(samples) else range(0, np.searchsorted(ptbins_c,2750))
    ptbins_c_plot = ptbins_c[plot_range]
    
    fig, ax = plt.subplots()
    xplot = np.geomspace(ptbins_c_plot.min() - (1), ptbins_c_plot.max(),1000)
    xplot2 = np.geomspace(ptbins_c_plot.min(), ptbins_c_plot.max(),1000)
    points_ls = []
    for flav in qfracs0.keys():
        lab = legend_dict_short[flav]
#         mc = next(ax._get_lines.prop_cycler)

        points = ax.errorbar(ptbins_c_plot, qfracs0[flav][plot_range, etaidx],
                             yerr=np.sqrt(qfrac_var0[flav][plot_range, etaidx]),
                             linestyle='none', label=lab,  **color_scheme[flav], capsize=1.6, capthick=0.7, linewidth=1.0)
        points2 = ax.errorbar(ptbins_c_plot, qfracs1[flav][plot_range, etaidx],
                              yerr=np.sqrt(qfrac_var1[flav][plot_range, etaidx]),
                              linestyle='none', mfc='none', markeredgewidth=1.2, **color_scheme[flav], capsize=1.6, capthick=0.7, linewidth=1.0)

        valid_fit_val = ~(np.isnan(qfracs1[flav]) | np.isinf(qfracs1[flav]) | (qfracs1[flav]==0))
        
#         ax.plot(xplot, spline0[flav](np.log10(xplot)), '--', markersize=0, **mc, linewidth=1.0)
#         sp1 = ax.plot(xplot, spline1[flav](np.log10(xplot)), '--', markersize=0, **mc, linewidth=1.0)
        ax.plot(xplot2, spline2D0[flav]((np.log10(xplot2), np.repeat([jeteta_bins.centres[etaidx]],len(xplot2)))),
                      '-.', markersize=0, **color_scheme[flav], linewidth=1.0)
        ax.plot(xplot2, spline2D1[flav]((np.log10(xplot2), np.repeat([jeteta_bins.centres[etaidx]],len(xplot2)))),
              '-.', markersize=0, **color_scheme[flav], linewidth=1.0)
# interp((np.log(np.arange(20,60,2)),[1]*20))
        if list(qfracs0.keys())[0] == flav:
            points_ls.append(points[0])
            points_ls.append(points2[0])
        
    
    ax.set_xscale('log')
    ax.set_xlabel('$p_{T,ptcl}$ (GeV)')
    ax.set_ylabel("Flavor fraction")
# fig.suptitle("Blaaah $x^2_5$")

    xlims = ax.get_xlim()

    ax.set_xticks([])
    ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    # ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    legend1 = ax.legend(points_ls, [samples[0], samples[1]], loc="upper left", bbox_to_anchor=(0.48, 1))
    leg2 = ax.legend(ncol=1, loc='upper right', bbox_to_anchor=(0.52, 1))
    ax.add_artist(legend1)
    # ax.add_artist(leg2)

    ylims = ax.get_ylim()
    ax.set_xlim(xlims)
    ax.set_ylim(ylims[0], ylims[1]*1.25)

    # ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
#     hep.cms.label("Something", data=False, year=2018, fontsize=15.6/1.25)
#     hep.cms.label(data=False, fontsize=15.6/1.25, loc=3)
    hep.label.exp_text(text=jeteta_bins.idx2plot_str(etaidx), loc=0)

    if saveplot:
        if not os.path.exists("fig/fractions"):
            os.mkdir("fig/fractions")

        fig_name = 'fig/fractions/fraction'+"".join(samples)
        print("Saving plot with the name = ", fig_name)
        plt.savefig(fig_name+'.pdf');
        plt.savefig(fig_name+'.png');

1;


def plot_spectra(histdict, labels, flav, etaidx, jeteta_bins, ptbins, saveplot=True, plotvspt=True):
    samples = list(histdict.keys())
    xbins = ptbins if plotvspt else jeteta_bins
    bins = jeteta_bins if plotvspt else ptbins
    xbins_c = xbins.centres
    xbins_ed = xbins.edges
    
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    if plotvspt:
        spectra = {samp: histdict[samp][flav][:,sum]/histdict[samp][flav].sum()['value'] for samp in samples}
    else:
        spectra = {samp: histdict[samp][flav][sum,:]/histdict[samp][flav].sum()['value'] for samp in samples}
    spectra['QCD-Py_weights'] = spectra['QCD-Py_weights']
#     pt_spectrumPy = histsPy[flav][:,sum]
#     pt_spectrumHer = histsHer[flav][:,sum]
#     ed = pt_spectrum.axes[0].edges
#     centres = pt_spectrum.axes[0].centers
    bin_widths = (xbins_ed[1:]-xbins_ed[:-1])
    colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for key in samples:
#         mc = next(ax._get_lines.prop_cycler)
#         colors
#         print(mc['color'])
        artist = (spectra[key]/bin_widths).plot1d(ax=ax, label=key, color=next(colors), linewidth=0.95) #, markersize=1.5) #color=plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(stack)])
        artist[0].errorbar[0].set_markersize(2.5)
#         artist[0].stairs.set_lw(0.95)
#     mc = next(ax._get_lines.prop_cycler)
#     (pt_spectrumHer/bin_widths).plot1d(ax=ax, color = mc['color'], label='QCD Py8')
    # lims = ax.get_xlim()
    # lims = [np.min(centres), np.max(centres)]
    if plotvspt:
        lims = [15,5000]
    else:
        lims = [-0.2,5.3]
    # ax.get_xlim()
    # ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    # ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(lims)
    ax.legend(labels)
    ax.set_xlim([0,6])

    denom = spectra[samples[0]].values()
    colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    artist = (spectra[samples[0]]/denom).plot1d(ax=ax2, color=next(colors), linewidth=0.95)
    artist[0].errorbar[0].set_markersize(2.5)
#     artist[0].stairs.set_lw(0.95)
#     assert False
    for key in samples[1:]:
        artist = (spectra[key]/denom).plot1d(ax=ax2, color=next(colors), linewidth=0.95)
        artist[0].errorbar[0].set_markersize(2.5)
#         artist[0].stairs.set_lw(0.95)
    if plotvspt:
        ax.set_xscale('log')
        ax2.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_ylim((0.0,2))
    if plotvspt:
        ax2.set_xticks([])
        ax2.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
        ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, numticks=30))  #numticks - the maximum number of ticks. 
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xticks(ax2.get_xticks())
    ax.set_xticklabels([])
    ax.set_xlim(lims)
    ax2.set_xlim(lims)
        
    ax2.hlines(1,-10, 10000, linestyles='--',color="black", 
       linewidth=1,)
    ax2.set_ylabel("ratio")
    ax2.set_xlabel('$p_{T,ptcl}$ (GeV)') if plotvspt else ax2.set_xlabel('$|\eta|$') 
    ax.set_ylabel("$dN/dp_{T,ptcl}$ (GeV)") if plotvspt else ax.set_ylabel("$dN/d|\eta|$")
#     hep.label.exp_text(text=f'{bins.idx2plot_str(eta_idx)}, {flav} jets', loc=2, ax=ax)
    hep.cms.label("Preliminary", loc=0, data=False, ax=ax)
    
    if saveplot:
        if not os.path.exists("fig/pt_spectra"):
            os.mkdir("fig/pt_spectra")

        fig_name = 'fig/pt_spectra/pt_spectra_'+"_".join(samples)
        print("Saving plot with the name = ", fig_name)
        plt.savefig(fig_name+'.pdf');
        plt.savefig(fig_name+'.png');

    plt.show()
1;

from uncertainty_helpers import get_ratio, read_data2

def poly4(x, *p):
    c0, c1, c2, c3, c4 = p
    xs = np.log10(x)
    res = c0+c1*xs+c2*xs**2+c3*xs**3+c4*xs**4
    return res

def poly4lims(x, xmin, xmax, *p):
    xcp = x.copy()
    lo_pos = xcp<xmin
    hi_pos = xcp>xmax
    xcp[lo_pos] = xmin
    xcp[hi_pos] = xmax
    return poly4(xcp, *p)

color_scheme2 = color_scheme.copy()
color_scheme2['QCD, 3 jets'] = {'color': 'brown', 'marker': 'o'}
color_scheme2['DY, 2 jets'] = {'color': 'cyan', 'marker': 'o'}

def plot_ratio_comparisons_samples(flav, etaidx, jeteta_bins, ptbins_c, eta_binning_str, correction_txt_dir, correction_txt, divide=False, inverse=False, use_recopt=False, plotsimfit=False):
    ''' Put ratio plots of many all flavors at the same place. Reproduce Fig. 31 in arXiv:1607.03663
    Output, polynomial coeficients of the data ratio fit
    divide: True if divide Herwig by Pythia, False if subtract
    inverse: True if plot corrections, False if plot responses
    use_recopt:  True if use reco pt, False if use gen pt
    '''        
    mean_name = "Median"
    mean_name_std = mean_name+'Std'

    ### Set plotting range (can be different from fitting range)
    start = np.searchsorted(ptbins_c, 16, side='left')
    end = 27


    #### Read median response/correction data
    denom_samples = ['_QCD-MG-Py', '_Pythia-TTBAR', '_DY-MG-Py'] #]s
    samples = ['_QCD-MG-Her', '_Herwig-TTBAR', '_DY-MG-Her'] #]
    sample_lab = ['QCD', 'TTBAR', 'DY'] #  ]
#     denom_samples = ['_QCD-MG-Py', '_QCD-MG-Py_leading_jets', '_Pythia-TTBAR', '_DY-MG-Py', '_DY-MG-Py_leading_jets'] #]
#     samples = ['_QCD-MG-Her', '_QCD-MG-Her_leading_jets', '_Herwig-TTBAR', '_DY-MG-Her', '_DY-MG-Her_leading_jets'] #]
#     sample_lab = ['QCD', r'QCD, 3 jets', 'TTBAR', 'DY', r'DY, 2 jets'] #  ]
    
    yvals = np.array([read_data2(mean_name, samp, flav, eta_binning_str)[start:end,etaidx] for samp in samples])
    stds  = np.array([read_data2(mean_name_std, samp, flav, eta_binning_str)[start:end,etaidx] for samp in samples])
    xvals = np.array([read_data2("MeanRecoPt", samp, flav, eta_binning_str)[start:end,etaidx] for samp in samples])
    
    yvals_d = np.array([read_data2(mean_name, samp, flav, eta_binning_str)[start:end,etaidx] for samp in denom_samples])
    stds_d  = np.array([read_data2(mean_name_std, samp, flav, eta_binning_str)[start:end,etaidx] for samp in denom_samples])
    xvals_d = np.array([read_data2("MeanRecoPt", samp, flav, eta_binning_str)[start:end,etaidx] for samp in denom_samples])
#     print('etaidx = ', etaidx)

    #### Read the fitted corrections
    corr_loc_Sum20_Py = [f"* * {correction_txt_dir}/{correction_txt}{eta_binning_str}.txt"]
    corr_loc_Sum20_Her = [f"* * {correction_txt_dir}/{correction_txt}_Her{eta_binning_str}.txt"]
    if plotsimfit:
        corr_loc_Sum20_Py_simfit = [f"* * {correction_txt_dir}/{correction_txt}_simfit{eta_binning_str}.txt"]
        corr_loc_Sum20_Her_simfit = [f"* * {correction_txt_dir}/{correction_txt}_simfit_Her{eta_binning_str}.txt"]
    ext = extractor()
    if plotsimfit:
        ext.add_weight_sets(corr_loc_Sum20_Py+corr_loc_Sum20_Her+corr_loc_Sum20_Py_simfit+corr_loc_Sum20_Her_simfit)
    else:
        ext.add_weight_sets(corr_loc_Sum20_Py+corr_loc_Sum20_Her)
    ext.finalize()
    evaluator = ext.make_evaluator()
        
    #### Clean and set up the data for plotting
    yvals[(yvals==0) | (np.abs(yvals)==np.inf)] = np.nan
    yvals_d[(yvals_d==0) | (np.abs(yvals_d)==np.inf)] = np.nan
    
    ratios = get_ratio(yvals, yvals_d, divide)
    if divide==True:
        ratio_unc = ((stds / yvals_d)**2 + (yvals/yvals_d**2 * stds_d)**2)**(1/2)
    else:
        ratio_unc = (stds**2+stds_d**2)**(1/2)
    
    if not use_recopt:
        xvals = ptbins_c[start:end]    
        

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        
    #### Plot the points
    for yval, std, samp in zip(ratios, ratio_unc, sample_lab):
        ax.errorbar(xvals, yval, yerr=std,
                    linestyle="none", label=samp, **color_scheme2[samp],
                    capsize=1.6, capthick=0.7, linewidth=1.0)
       
    #### Plot pre-fitted curves
    for fit_samp, lab in zip(['J', 'T'], ['QCD', 'TTBAR']):
        etaval = jeteta_bins.centres[etaidx]
        xvals_cont = np.geomspace(np.min(xvals), np.max(xvals), 100)
        yvals_cont = evaluator[f'{correction_txt}_Her{eta_binning_str}_{flav}{fit_samp}'](np.array([etaval]),xvals_cont)
        yvals_cont_d = evaluator[f'{correction_txt}{eta_binning_str}_{flav}{fit_samp}'](np.array([etaval]),xvals_cont)

        if inverse==True:
            yvals = 1/yvals
            yvals_d = 1/yvals_d
            ### Error propagation
            stds = yvals**2*stds
            stds_d = yvals_d**2*stds_d

        if inverse==False:
            yvals_cont = 1/yvals_cont
            yvals_cont_d = 1/yvals_cont_d

        ratios_cont = get_ratio(yvals_cont, yvals_cont_d, divide)
        ax.plot(xvals_cont, ratios_cont, markersize=0, **color_scheme[lab], label=lab+' fit')

    if plotsimfit:
        yvals_cont_simfit = evaluator[f'{correction_txt}_simfit_Her{eta_binning_str}_{flav}J'](np.array([etaval]),xvals_cont)
        yvals_cont_d_simfit = evaluator[f'{correction_txt}_simfit{eta_binning_str}_{flav}J'](np.array([etaval]),xvals_cont)
        yvals_cont_simfit = 1/yvals_cont_simfit
        yvals_cont_d_simfit = 1/yvals_cont_d_simfit
        ratios_cont_simfit = get_ratio(yvals_cont_simfit, yvals_cont_d_simfit, divide)
        ax.plot(xvals_cont, ratios_cont_simfit, markersize=0, label='simultaneous fit')

    ax.set_xscale('log')
    xlims = ax.get_xlim()
    
    ax.hlines(1,1, 10000, linestyles='--',color="black", linewidth=1,)
    
    ####################### Fit ####################
    fit_minx = np.searchsorted(ptbins_c, ptmin_global, side='left') - 1
    fit_maxx = np.searchsorted(ptbins_c, ptmax_global, side='left')
    
    xval4fit = np.tile(xvals[fit_minx:fit_maxx], len(sample_lab))
    yval4fit = np.concatenate(ratios[:,fit_minx:fit_maxx])
    ratio_unc4fit = np.concatenate(ratio_unc[:,fit_minx:fit_maxx])
    validpt_mask = ~(np.isnan(yval4fit) | np.isinf(yval4fit) | (yval4fit==0))
    xval4fit = xval4fit[validpt_mask]
    yval4fit = yval4fit[validpt_mask]
    ratio_unc4fit = ratio_unc4fit[validpt_mask]
    ### Put the minimum limit on the relative uncertainty to min_rel_uncert
    min_rel_uncert = 0.001
    if divide == True:
        where_limit_std = (ratio_unc4fit/yval4fit)<min_rel_uncert
        ratio_unc4fit[where_limit_std] = min_rel_uncert*yval4fit[where_limit_std]
    else:
        where_limit_std = ratio_unc4fit<min_rel_uncert
        ratio_unc4fit[where_limit_std] = min_rel_uncert
    
    p_poly4_1, arr = curve_fit(poly4, xval4fit, yval4fit, p0=[ 1, 1, 1, 1, 1])
    p_poly4, arr = curve_fit(poly4, xval4fit, yval4fit, p0=p_poly4_1, sigma=ratio_unc4fit)
#     p_poly4_1, arr = curve_fit(np.tile(xvals,len(sample_lab)), np.concatenate(ratios), means2fit, p0=[ 1, 1, 1, 1, 1])
    xfitmin = xval4fit.min()
    xfitmax = xval4fit.max()
    poly4fun = lambda x, p: poly4lims(x, xfitmin, xfitmax, *p)
    y_poly4 = poly4fun(xvals_cont, p_poly4)
    # y_poly4_now = poly4fun(xvals_cont, p_poly4_1)
    ax.plot(xvals_cont, y_poly4, label=r'Poly, n=4' ,linewidth=2.0, markersize=0);
    ####################### End fit ####################

    ####################### Calculate resonable limits excluding the few points with insane errors
    recalculate_limits=True
    if recalculate_limits:
        yerr_norm = np.concatenate(ratio_unc)
        y_norm = np.concatenate(ratios)
        norm_pos = (yerr_norm<0.01) &  (yerr_norm != np.inf) & (y_norm>-0.1)  
        if ~np.any(norm_pos):
            print("Cannot determine ylimits")
            norm_pos = np.ones(len(yerr_norm), dtype=int)
            raise Exception("Cannot determine ylimits")
        left_lim = np.min((y_norm-yerr_norm)[norm_pos])
        right_lim = np.max((yerr_norm+y_norm)[norm_pos])
        lim_pad = (right_lim - left_lim)/20
        ax.set_ylim(left_lim-lim_pad, right_lim+lim_pad*10)
    

    ####################### Formalities and save plot ####################`
    xlabel = r'$p_{T,reco}$ (GeV)' if use_recopt else r'$p_{T,ptcl}$ (GeV)'
    ax.set_xlabel(xlabel);
    ylab_pre = 'Her7/Py8' if divide else 'Her7-Py8'
    ylabel = r' (correction)' if inverse else r' (median response)'
    ax.set_ylabel(ylab_pre+ylabel);
    
    ax.set_xticks([10, 20, 50, 100, 500, 1000, 5000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    leg1 = ax.legend(ncol=1)
    ax.set_xlim(xlims)
    
    hep.label.exp_text(text=jeteta_bins.idx2plot_str(etaidx)+f', {flav} jets', loc=0)
    
    figdir = "fig/uncertainty"
    if not os.path.exists(figdir):
        os.mkdir(figdir)
    etastr = jeteta_bins.idx2str(etaidx)
    fig_name = f'fig/uncertainty/Pythia_Herwig_all_samples_{flav}_jets_{etastr}'
    print("Saving plot with the name = ", fig_name)
    plt.savefig(fig_name+'.pdf');
    plt.savefig(fig_name+'.png');
    plt.show()
    return [p_poly4, xfitmin, xfitmax]

def plot_uncertainty_antiflav(ptvals, etavals, HerPy_differences, additional_uncertainty_curves, uncertainties, ptoretastr, flavors, plotvspt=False):
    addc = additional_uncertainty_curves
    fig, ax = plt.subplots()

    xvals = ptvals if plotvspt else etavals
    flav_labs = []
    antiflav_labs = []

    for flav in flavors:
        lab = legend_dict_short[flav]
        norm_factor = 0               # no normalization for flal/antiflav uncertainty

        linestyle = '-.' if 'bar' in flav else '-'
        line = ax.plot(xvals, (addc[f'{flav}100']-norm_factor)*100, label=lab, markersize=0, linewidth=1.2, linestyle=linestyle,
                **color_scheme[flav])
        if 'bar' in flav:
            antiflav_labs.append(line[0])
        else:
            flav_labs.append(line[0])
    
    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1],color="gray",
        linewidth=1, alpha=0.4)

    legend1 = ax.legend(handles=antiflav_labs, loc='upper right', bbox_to_anchor=(0.52, 1), handlelength=1.5, title='antiflavor', title_fontsize=10)
    leg2 = ax.legend(handles=flav_labs, ncol=1, loc='upper left', bbox_to_anchor=(0.47, 1), handlelength=0.9, title='flavor' , title_fontsize=10)#, title='assembled\nfrom QCD', title_fontsize=10)
    ax.add_artist(legend1)
    xlabel = r'$p_{T}$ (GeV)' if plotvspt else r'$\eta$'
    ax.set_xlabel(xlabel);
    ylabel = 'JEC uncertainty (%)'
    ax.set_ylabel(ylabel);
    if plotvspt:
        ax.set_xscale('log')
        ax.set_xticks([10, 20, 50, 100, 500, 1000, 5000])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlim(15,1000)

    # ax.set_ylim(0.9885,1.0205)
    ylim_old = ax.get_ylim()
    ylim_pad = (ylim_old[1]-ylim_old[0])*0.4 if plotvspt else (ylim_old[1]-ylim_old[0])*0.62
    ax.set_ylim(ylim_old[0],ylim_old[1]+ylim_pad)
    labtxt = f'{ptoretastr}' #if plotvspt else f'{ptoretastr}'
#     labtxt = f'$\eta$ = {etabins_abs[ptoretaidx]}' if plotvspt else f'$p_T$ = {ptbins_c[ptoretaidx]} GeV'
    hep.label.exp_text(text=labtxt, loc=0)
    figdir = "fig/uncertainty"
    if not os.path.exists(figdir):
        os.mkdir(figdir)

    if plotvspt:
        fig_name = figdir+f"/JECuncertainty_vs_pt_eta_{ptoretastr}".replace('.','')
    else:
        fig_name = figdir+f"/JECuncertainty_vs_pt_pt_{ptoretastr}".replace('.','_')
    fig_name = fig_name.replace(', ', '_').replace(' ', '_').replace('$', '').replace('=', '_').replace('\eta', 'eta').replace('|', '').replace('<', '')
    print("Saving plot with the name = ", fig_name+".pdf / .png")
    plt.savefig(fig_name+'.pdf');
    plt.savefig(fig_name+'.png');
    plt.show()

def plot_uncertainty(ptvals, etavals, HerPy_differences, additional_uncertainty_curves, uncertainties, ptoretastr, flavors, plotvspt=False):
    addc = additional_uncertainty_curves
    fig, ax = plt.subplots()

    xvals = ptvals if plotvspt else etavals
    old_uncs = []
    for samp in ['QCD', 'DY']:    
        old_unc = ax.plot(xvals, (uncertainties[samp](etavals, ptvals)[:,0]-1)*100, '-.', markersize=0, linewidth=1.0,
                **color_scheme[samp], alpha=0.6)
        ax.plot(xvals, (HerPy_differences[samp][0]-addc['Rref'])*100, linestyle=(2, (4, 2)), label=samp, markersize=0,
                linewidth=1.2, **color_scheme[samp])
        old_uncs.append(old_unc[0])

    for flav in ['g', 'q', 'b', 'c']:
        color = color_scheme[flav] if flav!='q' else color_scheme['ud']
        old_unc = ax.plot(xvals, (uncertainties[flav](etavals, ptvals)[:,0]-1)*100, '-.', markersize=0, linewidth=1.0,
                **color, alpha=0.6)
        old_uncs.append(old_unc[0])

    for flav in flavors:
        lab = legend_dict[flav]
        ax.plot(xvals, (addc[f'{flav}100']-addc['Rref'])*100, label=lab, markersize=0, linewidth=1.2,
                **color_scheme[flav])
    
    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1],color="gray",
        linewidth=1, alpha=0.4)

    legend1 = ax.legend(old_uncs, ['']*len(old_uncs), loc='upper right', bbox_to_anchor=(0.52, 1), handlelength=1.5, title='Run 1', title_fontsize=10)
    leg2 = ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(0.47, 1), handlelength=0.9, title='Run 2' , title_fontsize=10)#, title='assembled\nfrom QCD', title_fontsize=10)
    ax.add_artist(legend1)
    xlabel = r'$p_{T}$ (GeV)' if plotvspt else r'$\eta$'
    ax.set_xlabel(xlabel);
    ylabel = 'JEC uncertainty (%)'
    ax.set_ylabel(ylabel);
    if plotvspt:
        ax.set_xscale('log')
        ax.set_xticks([10, 20, 50, 100, 500, 1000, 5000])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlim(15,1000)

    # ax.set_ylim(0.9885,1.0205)
    ylim_old = ax.get_ylim()
    ylim_pad = (ylim_old[1]-ylim_old[0])*0.4 if plotvspt else (ylim_old[1]-ylim_old[0])*0.62
    ax.set_ylim(ylim_old[0],ylim_old[1]+ylim_pad)
    labtxt = f'{ptoretastr}' #if plotvspt else f'{ptoretastr}'
#     labtxt = f'$\eta$ = {etabins_abs[ptoretaidx]}' if plotvspt else f'$p_T$ = {ptbins_c[ptoretaidx]} GeV'
    hep.label.exp_text(text=labtxt, loc=0)
    figdir = "fig/uncertainty"
    if not os.path.exists(figdir):
        os.mkdir(figdir)

    if plotvspt:
        fig_name = figdir+f"/JECuncertainty_vs_pt_eta_{ptoretastr}".replace('.','')
    else:
        fig_name = figdir+f"/JECuncertainty_vs_pt_pt_{ptoretastr}".replace('.','_')
    fig_name = fig_name.replace(', ', '_').replace(' ', '_').replace('$', '').replace('=', '_').replace('\eta', 'eta').replace('|', '').replace('<', '')
    print("Saving plot with the name = ", fig_name+".pdf / .png")
    plt.savefig(fig_name+'.pdf');
    plt.savefig(fig_name+'.png');
    plt.show()

def plot_HerPydiff(ptvals, HerPy_differences, additional_uncertainty_curves, divideHerPy, etaidx, jeteta_bins, pt_bins, pltstr2, flavors, combine_antiflavour):
    addc = additional_uncertainty_curves
    fig, ax = plt.subplots()

    lines = []
    markers = []
    for samp in ['QCD', 'DY', 'TTBAR']:    
#         mc = next(ax._get_lines.prop_cycler)
        line = ax.plot(ptvals, HerPy_differences[samp][0], linestyle=(0, (3.3, 2)), markersize=0, **color_scheme[samp], linewidth=1.2)
        marker = ax.errorbar(pt_bins.centres, HerPy_differences[samp][1], yerr=HerPy_differences[samp][2],
                        linestyle='none', **color_scheme[samp], capsize=1.6, capthick=0.7, linewidth=1.0)
        markers.append(marker[0])

        lines.append(line[0])
        
    pointsg20 = ax.plot(ptvals, addc['g20q80'], label='DY at 200 GeV', markersize=0, linewidth=1.2, **color_scheme["DY200"])
    for flav in flavors:
        linestyle = '-.' if 'bar' in flav else '-'
        if combine_antiflavour:
            lab = legend_dict[flav]
        else:
            lab = legend_dict_short[flav]
        ax.plot(ptvals, addc[f'{flav}100'], label=lab, markersize=0, linewidth=1.2, linestyle=linestyle, **color_scheme[flav])

    vlinecoord = 1 if divideHerPy else 0
    ax.hlines(vlinecoord ,1, 10000,color="gray",
        linewidth=1, alpha=0.4)

    ax.hlines(addc['g20q80_fixed'], 1, 10000, linestyles='--',color=color_scheme["DY200"]['color'],
        linewidth=1, alpha=0.9)

    leg1_handles = [(ai,bi) for ai, bi, in zip(lines,markers)]
    legend1 = ax.legend(leg1_handles, ['QCD', 'DY', 'TTBAR'], loc="upper right", bbox_to_anchor=(0.52, 1), handlelength=1.5) # seg.len=5) #, title='correction', title_fontsize=10)
#     assert False
    leg2 = ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(0.48, 1))#, title='assembled\nfrom QCD', title_fontsize=10)
    ax.add_artist(legend1)
    xlabel = r'$p_{T}$ (GeV)'
    ax.set_xlabel(xlabel);
    ylab_pre = 'Her7/Py8' if divideHerPy else 'Her7-Py8'
    ylabel = r' (correction)'
    ax.set_ylabel(ylab_pre+ylabel);
    ax.set_xscale('log')
    ax.set_xticks([10, 20, 50, 100, 500, 1000, 5000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    # ax.set_ylim(0.9885,1.0205)
    ax.set_xlim(15,1000)
    ylim_old = ax.get_ylim()
    ylim_pad = (ylim_old[1]-ylim_old[0])*0.3
    ax.set_ylim(ylim_old[0],ylim_old[1]+ylim_pad)

    hep.label.exp_text(text=jeteta_bins.idx2plot_str(etaidx)+pltstr2, loc=0)

    figdir = "fig/uncertainty"
    if not os.path.exists(figdir):
        os.mkdir(figdir)
    pltstr2 = pltstr2.replace(', ', '_').replace(' ', '_').replace('$', '').replace('=', '_').replace('\eta', 'eta').replace('|', '').replace('<', '')
    add_name = '/Herwig_Pythia_ratio' if divideHerPy else '/Herwig_Pythia_difference'
    fig_name = figdir+add_name+pltstr2+'_'+jeteta_bins.idx2str(etaidx)
    print("Saving plot with the name = ", fig_name+".pdf / .png")
    plt.savefig(fig_name+'.pdf');
    plt.savefig(fig_name+'.png');
    plt.show()


def plot_all_flavor_comparison(num_sample_name,
                         denom_sample_name, jeteta_bins, ptbins_c, eta_binning_str, fit_samp='J', etaidx=0):
    ''' Put ratio plots of many all flavors at the same place. Reproduce Fig. 31 in arXiv:1607.03663
    '''

    inverse=False   #True if plot corrections, False if plot responses
    use_recopt=False   #True if use reco pt, False if use gen pt
    flavors = ['g', 'q' ,'c', 'b'] #, 'unmatched']
    
    mean_name = "Median"
    mean_name_std = mean_name+'Std'
    start = np.searchsorted(ptbins, 15, side='left')
#     etaidx = np.searchsorted(jeteta_bins_abs, 0, side='left')
    
    yvals = np.array([read_data2(mean_name, num_sample_name, flav, eta_binning_str)[start:,etaidx] for flav in flavors])
    stds  = np.array([read_data2(mean_name_std, num_sample_name, flav, eta_binning_str)[start:,etaidx] for flav in flavors])
    xvals = np.array([read_data2("MeanRecoPt", num_sample_name, flav, eta_binning_str)[start:,etaidx] for flav in flavors])
    
    yvals_d = np.array([read_data2(mean_name, denom_sample_name, flav, eta_binning_str)[start:,etaidx] for flav in flavors])
    stds_d  = np.array([read_data2(mean_name_std, denom_sample_name, flav, eta_binning_str)[start:,etaidx] for flav in flavors])
    xvals_d = np.array([read_data2("MeanRecoPt", denom_sample_name, flav, eta_binning_str)[start:,etaidx] for flav in flavors])
#     print('etaidx = ', etaidx)

    corr_loc_Sum20_Py = [f"* * Summer20UL18_V2_MC/Summer20UL18_V2_MC_L5Flavor_AK4PFchs{eta_binning_str}.txt"]
    corr_loc_Sum20_Her = [f"* * Summer20UL18_V2_MC/Summer20UL18_V2_MC_L5Flavor_AK4PFchs_Her{eta_binning_str}.txt"]
    ext = extractor()
    ext.add_weight_sets(corr_loc_Sum20_Py+corr_loc_Sum20_Her)
    ext.finalize()
    evaluator = ext.make_evaluator()
        
    yvals[(yvals==0) | (np.abs(yvals)==np.inf)] = np.nan
    yvals_d[(yvals_d==0) | (np.abs(yvals_d)==np.inf)] = np.nan
    
    ratios = yvals/yvals_d
    ratio_unc = ((stds / yvals_d)**2 + (yvals/yvals_d**2 * stds_d)**2)**(1/2)
    
    if not use_recopt:
        xvals = ptbins_c[start:]    
        
    etaval = jeteta_bins.centres[etaidx]
    xvals_cont = np.geomspace(np.min(xvals), np.max(xvals), 100)
    yvals_cont = np.array([evaluator[f'Summer20UL18_V2_MC_L5Flavor_AK4PFchs_Her{eta_binning_str}_{flav}{fit_samp}'](np.array([etaval]),xvals_cont)
                           for flav in flavors])
    yvals_cont_d = np.array([evaluator[f'Summer20UL18_V2_MC_L5Flavor_AK4PFchs{eta_binning_str}_{flav}{fit_samp}'](np.array([etaval]),xvals_cont)
                       for flav in flavors])
    if inverse==True:
        yvals = 1/yvals
        yvals_d = 1/yvals_d
        ### Error propagation
        stds = yvals**2*stds
        stds_d = yvals_d**2*stds_d
        
    if inverse==False:
        yvals_cont = 1/yvals_cont
        yvals_cont_d = 1/yvals_cont_d
    
    
    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        
#     assert False
    for yval, std, flav in zip(ratios, ratio_unc, flavors):
        ax.errorbar(xvals, yval, yerr=std,
                    linestyle="none", label=legend_dict[flav], **color_scheme[flav],
                    capsize=1.6, capthick=0.7, linewidth=1.0)
#         assert not lab == 'unmatched'
       
    ratios_cont = yvals_cont/yvals_cont_d
#     ax.set_prop_cycle(None)
    for yval, flav in zip(ratios_cont, flavors):
        ax.plot(xvals_cont, yval, markersize=0, **color_scheme[flav])
    
    ax.set_xscale('log')
    xlims = ax.get_xlim()
    
    ax.hlines(1,1, 10000, linestyles='--',color="black", linewidth=1,)
    ######################## Calculate resonable limits excluding the few points with insane errors
    recalculate_limits=True
    if recalculate_limits:
        yerr_norm = np.concatenate(ratio_unc)
        y_norm = np.concatenate(ratios)
        norm_pos = (yerr_norm<0.01) &  (yerr_norm != np.inf) & (y_norm>-0.1)  
        if ~np.any(norm_pos):
            print("Cannot determine ylimits")
            norm_pos = np.ones(len(yerr_norm), dtype=int)
            raise Exception("Cannot determine ylimits")
        left_lim = np.min((y_norm-yerr_norm)[norm_pos])
        right_lim = np.max((yerr_norm+y_norm)[norm_pos])
        lim_pad = (right_lim - left_lim)/20
        ax.set_ylim(left_lim-lim_pad, right_lim+lim_pad*8)
    
    xlabel = r'$p_{T,reco}$ (GeV)' if use_recopt else r'$p_{T,ptcl}$ (GeV)'
    ax.set_xlabel(xlabel);
    ylab_pre = 'Her7/Py8' 
    ylabel = r' (correction)' if inverse else r' (median response)'
    ax.set_ylabel(ylab_pre+ylabel);
    
    ax.set_xticks([10, 20, 50, 100, 500, 1000, 5000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    leg1 = ax.legend(ncol=1)
    ax.set_xlim(xlims)
    
    title_name = 'QCD' if fit_samp=='J' else 'ttbar'
    hep.label.exp_text(text=jeteta_bins.idx2plot_str(eta_idx)+f', {title_name}', loc=0)
    
    etastr = jeteta_bins.idx2str(eta_idx)
    fig_name = f'fig/uncertainty/Pythia_Herwig_ratio_{etastr}_using_{fit_samp}_fits'
    print("Saving plot with the name = ", fig_name)
    plt.savefig(fig_name+'.pdf');
    plt.savefig(fig_name+'.png');
    plt.show();