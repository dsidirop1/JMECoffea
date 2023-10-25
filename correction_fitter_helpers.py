header_txt = ('# L5 flavor corrections for IC5 algorithm \n'+
'# [gJ] (gluons from diJet mixture) \n'+
'# [bJ] (b quark from diJet mixture) \n'+
'# [cJ] (c quark from diJet mixture) \n'+
'# [qJ] (uds quarks from diJet mixture) \n'+
'# [udJ] (ud quark from diJet mixture) \n'+
'# [uJ] (u quark from diJet mixture) \n'+
'# [dJ] (d quark from diJet mixture) \n'+
'# [gT] (gluons from ttbar events) \n'+
'# [qT] (uds quarks from ttbar events) \n'+
'# [cT] (c quark from ttbar events) \n'+
'# [bT] (b quark from ttbar events) \n'+
'# energy mapping: ptGen = (pt - p5)/p6 \n'+
'# parametrization: p2+p3*logPt+p4*logPt^2, constant if Pt<p0 or Pt>p1 \n'+
'#etamin  etamax  #ofparameters  ptmin  ptmax    p2         p3        p4     mapping: p5        p6 ')

def save_correction_txt_file(txtfile_outname, fit_res_all_tags):
    '''
    Saves the corrections in the txt file with the name `txtfile_outname`.
    fit_res_all_tags: for each correction tag (e.g., T and J), a dictionary of corrections for each flavor
    '''
    with open(txtfile_outname, 'w') as file:
        file.write(header_txt+'\n')
        for tag in fit_res_all_tags:
            fit_res_all = fit_res_all_tags[tag]
            for key in fit_res_all.keys():
                fit_res = fit_res_all[key]
                file.write(f'[{key}]\n')
                file.write('{1 JetEta 1 JetPt ([0]+[1]*log10(x)+[2]*pow(log10(x),2)+[3]*pow(log10(x),3)+[4]*pow(log10(x),4)) Correction L5Flavor}\n')
                ### copy from the positive eta region into the negative
                fit_res = np.vstack([np.hstack([np.flip(fit_res[:,0:2]*-1), np.flip(fit_res[:,2:], 0)]), fit_res])
                for row in fit_res:
                    row[2] = row[2]+2  #+2 because of the pt lower/higher limits that are not accounted into the # parameters before
                    line2write = ('{:>11} '*5+'{:>13} '*(int(row[2])-2)).format(*row[:2], int(row[2]), *np.round(row[3:], 7))+'\n'
                    file.write(line2write);
    print("Saving the corrections with the name = ", txtfile_outname)

# from scipy.optimize import fsolve
from scipy.optimize import brentq
# scipy.optimize.brentq

def find_stationary_pnt_poly(xmin, xmax, *p, degree=4):
    '''Finds the last x within the limits [xmin, xmax], where the derivative of the n-th order polynomial with
    coefficients p, changes sign. n is allowed to be 4 or 3 at the moment. If there is no such point in outputs xmax.  '''
    if degree not in [3,4]:
        raise ValueError(f"Degree can be either 3 or 4. The value given is {degree}")
    if degree==3:
        c0, c1, c2, c3 = p
    elif degree==4:
        c0, c1, c2, c3, c4 = p
    xmin = np.log10(xmin)
    xmax = np.log10(xmax)
    xs = np.linspace(xmin, xmax, 1000)
    if degree==3:
        deriv = lambda xs: c1+2*c2*xs+c3*3*xs**2
    elif degree==4:
        deriv = lambda xs: c1+2*c2*xs+c3*3*xs**2+4*c4*xs**3
    signx = np.sign(deriv(xs))
    changes_sign = signx[1:]*signx[:-1]
    last_change_idx = np.where(changes_sign==-1)[0]
    if len(last_change_idx)==0:
        last_change_idx = 0
        root = xmax
    else:
        last_change_idx = last_change_idx[-1]
        root = brentq(deriv, xs[last_change_idx], xs[last_change_idx+1] )
    
    return 10**root

#### initial values borrowed from Winter14 data
#### https://github.com/cms-jet/JECDatabase/blob/master/textFiles/Winter14_V8_MC/Winter14_V8_MC_L5Flavor_AK5Calo.txt/
#### used for fit `response_fnc`
init_vals_2014 = {
    'b':
    [[0.540002, 13.8495, 17.8549, -0.215711, 0.576285, 1.42258],
    [0.73119, 7.52866, 17.3681, -0.078402, 1.21665, 1.69878],
    [0.999952, 0.0322738, -1.05606, -19.6994, 0.720321, -1.58314],
    [0.135913, 7.92441, 3.85698, -0.804604, 1.11911, 0.732041]],
    'c' :
    [[ 0.940259, 0.705481, 0.23917, -0.826926, 0.311473, -0.514041],
    [0.982083, 0.238007, 4.35924, -0.0314618, 5.91028, 1.67749],
    [0.733505, 7.26794, 12.2028, -0.756302, 0.0895257, -1.96324],
    [0.932305, 1.15954, 17.1731, -0.471313, 2.58424, 0.254917]],
    'g' :
    [[0.877892, 3.10194, 1.16568, -677.876, 0.0325026, -12.9485],
    [0.983775, 0.247943, 1.55373, -0.0254802, 3.35748, 1.71263],
    [-0.972548, 38.8683, 2.47151, -44.0233, 0.0901665, -3.15495],
    [1.0655, -0.0680325, -0.509038, -8.59434e+06, 42.6162, 0.357177]],
    'd':
    [[1.28488, -46.3648, 151.749, -0.0108461, 15.4256, 1.63377],
    [ 1.50931, -118.71, 224.19, -0.0196468, 4.62655, 1.51581],
    [0.692016, 8.26488, 11.1655, -0.802769, 0.116182, -1.16094],
    [1.01244, -0.0926519, -0.12138, -3.69494e+07, 7.15634, -0.625288]],  
    'u':
    [[1.28488, -46.3648, 151.749, -0.0108461, 15.4256, 1.63377],
    [ 1.50931, -118.71, 224.19, -0.0196468, 4.62655, 1.51581],
    [0.692016, 8.26488, 11.1655, -0.802769, 0.116182, -1.16094],
    [1.01244, -0.0926519, -0.12138, -3.69494e+07, 7.15634, -0.625288]],  
    's':
    [[1.28488, -46.3648, 151.749, -0.0108461, 15.4256, 1.63377],
    [ 1.50931, -118.71, 224.19, -0.0196468, 4.62655, 1.51581],
    [0.692016, 8.26488, 11.1655, -0.802769, 0.116182, -1.16094],
    [1.01244, -0.0926519, -0.12138, -3.69494e+07, 7.15634, -0.625288]],
    'all':
    [[0.540002, 13.8495, 17.8549, -0.215711, 0.576285, 1.42258],
    [0.73119, 7.52866, 17.3681, -0.078402, 1.21665, 1.69878],
    [0.999952, 0.0322738, -1.05606, -19.6994, 0.720321, -1.58314],
    [0.135913, 7.92441, 3.85698, -0.804604, 1.11911, 0.732041]],  
    'ud':
    [[1.28488, -46.3648, 151.749, -0.0108461, 15.4256, 1.63377],
    [ 1.50931, -118.71, 224.19, -0.0196468, 4.62655, 1.51581],
    [0.692016, 8.26488, 11.1655, -0.802769, 0.116182, -1.16094],
    [1.01244, -0.0926519, -0.12138, -3.69494e+07, 7.15634, -0.625288]],  
    'q':
    [[1.28488, -46.3648, 151.749, -0.0108461, 15.4256, 1.63377],
    [ 1.50931, -118.71, 224.19, -0.0196468, 4.62655, 1.51581],
    [0.692016, 8.26488, 11.1655, -0.802769, 0.116182, -1.16094],
    [1.01244, -0.0926519, -0.12138, -3.69494e+07, 7.15634, -0.625288]],
    
}

## Initial values for the two gaussian fit
init_two_gaus = [3,0,1,2,0,1,2,3,4]
# Better starting fit values I found
init_vals_2014['b'][0] = [ 9.81014871e-01, -6.46744813e-03, -1.05658840e+00,  5.35445486e+03, 2.99200015e+01,  1.21399356e+02]
init_vals_2014['b'][3] = [ 9.81014871e-01, -6.46744813e-03, -1.05658840e+00,  5.35445486e+03, 2.99200015e+01,  1.21399356e+02]


from JetEtaBins import JetEtaBins, PtBins
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

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

def poly3(x, *p):
    c0, c1, c2, c3 = p
    xs = np.log10(x)
    res = c0+c1*xs+c2*xs**2+c3*xs**3
    return res

def poly3lims(x, xmin, xmax, *p):
    xcp = x.copy()
    lo_pos = xcp<xmin
    hi_pos = xcp>xmax
    xcp[lo_pos] = xmin
    xcp[hi_pos] = xmax
    return poly3(xcp, *p)

def response_fnc(x, *p):
    p0, p1, p2, p3, p4, p5 = p
    logx = np.log10(x)
    return p0+(p1/((logx**2)+p2)) + (p3*np.exp(-p4*((logx-p5)*(logx-p5))))

def two_gaus_fnc(x, *p):
    p0, p1, p2, p3, p4, p5, p6, p7, p8 = p
    return (  p0
            + (p1/((np.log10(x)**2)+p2))
            + (p3*np.exp(-p4*((np.log10(x)-p5)*(np.log10(x)-p5))))
            + (p6*np.exp(-p7*((np.log10(x)-p8)*(np.log10(x)-p8))))
           )

def response_fnc_raw(x, p0, p1, p2, p3, p4, p5):
    response_fnc(x, *[p0, p1, p2, p3, p4, p5])

from scipy.stats import chi2
import mplhep as hep
from fileNames.available_datasets import legend_labels
figdir = "fig/median_correction_fits"


fits2plot = { ##name: [function, initial values, # parameters]
             "MC truth": [response_fnc, init_vals_2014, 6],
             "MC truth extended": [two_gaus_fnc, init_two_gaus, 9],
             "Poly, n=4": [poly4, [ 1, 1, 1, 1, 1], 5],
             "Poly, n=3": [poly3, [ 1, 1, 1, 1], 4],
             }

main_fit = "Poly, n=4"

def fit_corrections(etaidx, data_dict, flav, data_tags,
                           fits2plot, main_fit,
                    figdir2=figdir+'correction_fit',
                    jeteta_bins=JetEtaBins(), pt_bins=PtBins(),
                    plot_initial_val=False,
                    use_recopt=True, maxlimit_static_pnt=True, max_ptval=None,
                    min_rel_uncert=0.0,
                    min_rel_uncert_relative=0,
                    show_original_errorbars=False,
                    ncoefs_out=5, ):
    """ fit the data and plot
    """
    ###################### Logistics with the input ######################
    keys = [key for key in data_dict.keys()]
    if maxlimit_static_pnt:
        if max_ptval==None:
            max_ptval=5000
        else:
            raise ValueError(f"Remove `max_ptval` if using `maxlimit_static_pnt`. `max_ptval` given as {max_ptval} but `maxlimit_static_pnt` is set to {maxlimit_static_pnt}.")
    elif max_ptval==None:
        max_ptval=500

    ### pt limits for the fit
    ptmin_idx = np.searchsorted(pt_bins.centres, 30, side='left')-1 #-1: so that the first bin includes the value
    ptmax_idx = np.searchsorted(pt_bins.centres, max_ptval, side='right')
    data_range = tuple([range(ptmin_idx,ptmax_idx), etaidx])

    yvals = np.array([data_dict[key][0][data_range] for key in keys])
    stds  = np.array([data_dict[key][1][data_range] for key in keys])
    reco_pts  = np.array([data_dict[key][2][data_range] if len(data_dict[key][2].shape)==2 else data_dict[key][2][data_range] for key in keys])

    validpt_mask = ~(np.isnan(yvals) | np.isinf(yvals) | (yvals==0))

    xvals = reco_pts if use_recopt else np.array([pt_bins.centres[data_range]]*len(yvals))

    ### Put the minimum limit on the relative uncertainty to min_rel_uncert
    # only the first case makes sence here imo, because it defines the minimum uncertainty relative to the range
    # but the second case kept for a while to be consistent with the not
    if min_rel_uncert<=0:
        min_rel_uncert_tmp = min_rel_uncert_relative*(np.nanmax(yvals)-np.nanmin(yvals))
    else:
        min_rel_uncert_tmp = min_rel_uncert
    stds_orig = stds.copy()
    where_limit_std = (stds/yvals)<min_rel_uncert_tmp
    stds[where_limit_std] = min_rel_uncert_tmp*yvals[where_limit_std]


    if np.sum(validpt_mask)==0:
        fit_res_new = np.concatenate([[jeteta_bins.edges[etaidx], jeteta_bins.edges[etaidx+1],
                               ncoefs_out, 
                               pt_bins.centres[ptmin_idx], pt_bins.centres[ptmax_idx-1]],
                              [1,0,0,0,0] ])
        print("No points to fit. Returning a unity function.")
        return fit_res_new

    means2fit = yvals[validpt_mask]
    means_unc2fit = np.abs(stds[validpt_mask])
    ptbins2fit = xvals[validpt_mask]
    fit_min_lim = np.min(ptbins2fit)
    fit_max_lim = np.max(ptbins2fit)

    xplot = np.linspace(fit_min_lim, fit_max_lim, 1000)

    ###################### Fits ######################
    
    fitres = {}
    for fit in fits2plot:
        if len(means2fit)>(fits2plot[fit][2]+2):
            try:
                init_vals = fits2plot[fit][1]
                if type(init_vals) is dict:
                    init_vals = init_vals[flav][etaidx]
                p, arr = curve_fit(fits2plot[fit][0], ptbins2fit, means2fit, p0=init_vals)
                p_err, arr = curve_fit(fits2plot[fit][0], ptbins2fit, means2fit, p0=p, sigma=means_unc2fit)
            except(RuntimeError):
                print(f"{fit} fit failed")
                p, p_err = [[np.nan]*fits2plot[fit][2]]*2
            # except(TypeError):
        else:
            print(f"Too little points for {fit} fit")
            p, p_err = [[np.nan]*fits2plot[fit][2]]*2
        fitres[fit] = p_err
    
    chi2s = {fit: np.sum((fits2plot[fit][0](ptbins2fit, *fitres[fit]) - means2fit)**2/means_unc2fit**2)
                 for fit in fits2plot}
    Ndofs = {fit: len(ptbins2fit) - fits2plot[fit][2] for fit in fits2plot}

    ## if chi2 of n=3 polynomial outside the 2 one-sided std of the chi2 distribution, use n=3 polynomial.
    if main_fit == "Poly, n=4" and "Poly, n=3" in fits2plot and chi2.ppf(1-0.158, Ndofs["Poly, n=3"])>chi2s["Poly, n=3"]:
        main_fit == "Poly, n=3"
    
    print(f"Using the {main_fit} fit results ")
    if maxlimit_static_pnt:
        if main_fit=="Poly, n=3":
            fit_degree = 3 
        elif main_fit=="Poly, n=4":
            fit_degree = 4
        else:
            ValueError(f"Main fit is {main_fit} but the derivative for the static point is not defined for this fit.")
        fit_max_lim_new = find_stationary_pnt_poly(fit_min_lim, fit_max_lim, *fitres[main_fit], degree=fit_degree)
    else:
        fit_max_lim_new = fit_max_lim
    fit_max_lim_idx = np.searchsorted(np.sort(ptbins2fit), fit_max_lim_new)
    if maxlimit_static_pnt & (fit_max_lim_idx==len(ptbins2fit)) | (fit_max_lim_idx<=len(ptbins2fit)-6):
        # static point is too low or the last point that usually fluctuates out
        fit_max_lim_idx = len(ptbins2fit)-3
        fit_max_lim_new = np.sort(ptbins2fit)[fit_max_lim_idx]
    xplot_max_new = np.searchsorted(xplot, fit_max_lim_new)

    main_fit_res = fitres[main_fit]
    if ncoefs_out<len(main_fit_res):
        raise ValueError(f"ncoefs is smaller than the number of coefficients of the main fit."
                        +"ncoefs_out={ncoefs_out}, len(main_fit_res)={len(main_fit_res)}."
                        +"Either raise the number of coefficients of the output or choose a different output function.")
    fit_res_new = np.concatenate([[jeteta_bins.edges[etaidx], jeteta_bins.edges[etaidx+1],
                                ncoefs_out, 
                                fit_min_lim, fit_max_lim_new],
                                np.pad(main_fit_res, (0, ncoefs_out-len(main_fit_res))) ])

    curve_yvals = {fit: fits2plot[fit][0](xplot, *fitres[fit]) for fit in fits2plot}


    ###################### Plotting ######################
    fig, ax = plt.subplots()
    for yval, std, reco_pt, std_orig, data_tag in zip(yvals, stds, xvals, stds_orig, data_tags):
        plt.errorbar(reco_pt, yval, yerr=std, marker='o',
                    linestyle="none", label=data_tag, #{jeteta_bins.idx2plot_str(etaidx)}',
                    capsize=1.7, capthick=0.9, linewidth=1.0)
        if show_original_errorbars:
            plt.errorbar(reco_pt, yval, yerr=std_orig, marker='o',
                linestyle="none", label=f'Original errorbars',
                capsize=1.5, capthick=0.7, linewidth=0.9, markersize=0, color='black') #, color='blue')
    
    for fit in fits2plot:
        if np.isnan(chi2s[fit]): 
            lab = f'{fit} failed'
        else:
            polytxt3 = ', selected' if fit==main_fit and len(fits2plot)>1 else ''
            lab= fit+r'; $\chi^2/N_{dof} = $'+r' {0:0.3g}/{1:0.0f}'.format(chi2s[fit], Ndofs[fit])+polytxt3
        ax.plot(xplot, curve_yvals[fit], label=lab, markersize=0);
        if maxlimit_static_pnt and fit==main_fit:
            ax.plot(xplot[:xplot_max_new], curve_yvals[fit][:xplot_max_new], label=f'{fit}; range chosen', markersize=0, linewidth=0.8); #
        if plot_initial_val and fit=="MC truth":
            yvals_init = response_fnc(xplot, *fits2plot[fit][1])
            ax.plot(xplot, yvals_init, label=f"Initial values for {fit}", markersize=0);
        
    ###################### Plot formalities ######################
    ylim_tmp = ax.get_ylim()
    ylim_pad = (ylim_tmp[1] - ylim_tmp[0])/1.6
    ax.set_ylim(ylim_tmp[0], ylim_tmp[1]+ylim_pad)

    ax.set_xlabel(r'$p_{T,reco}$ (GeV)')
    ax.set_ylabel(r'correction (1/median)');
    ax.set_xscale('log')

    ax.set_xticks([])
    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    # hep.cms.label("Preliminary", loc=0, data=False, ax=ax)
    hep.cms.label("Private work", loc=0, data=False, ax=ax, rlabel='')
    hep.label.exp_text(text=f'{jeteta_bins.idx2plot_str(etaidx)}; {flav} jets', loc=2, fontsize=mpl.rcParams["font.size"]/1.15)

    ### hack put errorbars before the curves in the legend
    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    #specify order of items in legend
    nfits = len(fits2plot)+maxlimit_static_pnt
    order = np.concatenate([np.arange(nfits,nfits+len(keys)+show_original_errorbars), np.arange(nfits)]) 

    #add legend to plot
    odered_handles = [handles[idx] for idx in order]
    ordered_labels = [labels[idx] for idx in order]
 
    ax.legend(odered_handles,
              ordered_labels,
              loc="upper left", bbox_to_anchor=(0.01, 0.90)) #, prop={'size': 10}
    
    figdir2 = (figdir+'/'+data_tag.replace(legend_labels["ttbar"]["lab"], 'ttbar').replace(', ', '-')
                .replace(" ", "_").replace("+", "_").replace('(', '').replace(')', '').replace('/', '').replace('\n', '')
              )
    if not os.path.exists(figdir2):
        os.mkdir(figdir2)
        
    add_name = f'correction_fit_{flav}_'+jeteta_bins.idx2str(etaidx)
    fig_name = figdir2+'/'+add_name
        
    print("Saving plot with the name = ", fig_name+".pdf / .png")
    plt.savefig(fig_name+'.pdf');
    plt.savefig(fig_name+'.png');

    plt.show();
    plt.close();
    
    return fit_res_new