import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from cycler import cycler

# colour bling colors = [#00429d,#394c99,#545793,#686867,#737373,#df4639,#d76132,#ce7628,#c3891b,#b89a00]
colourblind_cols = ["#920000", "#006ddb", "#24ff24","#db6d00",
                    "#004949","#b66dff","#b6dbff","#924900",
                    "#ffff6d", "#490092","#009292","#ff6db6",
                    "#000000","#ffb6db", "#6db6ff" ]
cols = ['#a6cee3', '#e31a1c','#1f78b4','#6a3d9a','#33a02c',
                            '#fb9a99', '#fdbf6f','#b2df8a','#ff7f00',
                            '#cab2d6','#ffff99','#b15928']

oldcols = ['#4C72B0', '#55A868', '#C44E52',
                    '#8172B2', '#CCB974', '#64B5CD',
                    '#e377c2', '#7f7f7f',
                    '#bcbd22', '#17becf']

def pltStyle(style='paper'):
    plt.style.use('default')
    plt.style.use('seaborn-paper')
    cols_cyc = cycler(color=colourblind_cols)
    markers_cyc = cycler(marker=['o','s', 'p','d', 'D', 'H', '^']*2+['v'])

    if style=='hep':
        import mplhep as hep
        hep.style.use("CMS")
        plt.rc('font', family='serif')

        ### make the size of the plot size_frac x smaller, but keep all the sizes relativelly the same
        size_frac = 2.5
        hep_parms = plt.rcParams
        plt.rcParams['figure.figsize'] = [hep_parms['figure.figsize'][i]/size_frac for i in range(2)]
        plt.rcParams['font.size'] = hep_parms['font.size']/size_frac*1.15
        plt.rcParams['xtick.major.size'] = plt.rcParams['ytick.major.size'] = hep_parms['xtick.major.size']/size_frac
        plt.rcParams['xtick.minor.size'] = plt.rcParams['ytick.minor.size'] = hep_parms['xtick.minor.size']/size_frac
        plt.rcParams['legend.borderpad'] = hep_parms['legend.borderpad']/size_frac
        plt.rcParams['legend.handlelength'] = hep_parms['legend.handlelength']/size_frac
        plt.rcParams['xtick.major.pad'] = plt.rcParams['ytick.major.pad'] = hep_parms['xtick.major.pad']/size_frac
        plt.rcParams['axes.labelpad'] = hep_parms['axes.labelpad']/size_frac
        plt.rcParams['lines.markersize'] = hep_parms['lines.markersize']/size_frac*2
        plt.rcParams['xtick.major.width'] = hep_parms['xtick.major.width']/size_frac*2
        plt.rcParams['xtick.minor.width'] = hep_parms['xtick.minor.width']/size_frac*2
        plt.rcParams['ytick.major.width'] = hep_parms['ytick.major.width']/size_frac*2
        plt.rcParams['ytick.minor.width'] = hep_parms['ytick.minor.width']/size_frac*2
        plt.rcParams['axes.linewidth'] = hep_parms['axes.linewidth']/size_frac
        plt.rcParams['legend.labelspacing'] = hep_parms['legend.labelspacing']/size_frac
        plt.rcParams['legend.columnspacing'] =  hep_parms['legend.columnspacing']/size_frac

        plt.rcParams['lines.markeredgewidth']=0
        plt.rcParams['axes.prop_cycle'] = cols_cyc+markers_cyc


    elif style=='paper':
        ### For paper - one smaller image full page

        plt.rcParams['figure.figsize'] = [4.2, 2.8]  #Latex text width =0.7*page width = 8.2 inches
#        plt.rcParams['figure.subplot.wspace'] = 0.33
        plt.rcParams['figure.subplot.top'] = 0.87
        plt.rcParams['figure.subplot.bottom'] = 0.16
        plt.rcParams['figure.subplot.right'] = 0.97
        plt.rcParams['figure.subplot.left'] = 0.162
        plt.rcParams['axes.prop_cycle'] = cols_cyc

    elif style=='presentation-square':
        ### For paper - one smaller image half a page
        plt.rcParams['figure.figsize'] = [3.7, 3.5]  #Latex text width =0.7*page width = 8.2 inches
#         plt.rcParams['figure.subplot.top'] = 0.97 #if don't add titles to plots
        plt.rcParams['figure.subplot.top'] = 0.91  #if add titles to plots
        plt.rcParams['figure.subplot.bottom'] = 0.13
        plt.rcParams['figure.subplot.right'] = 0.97
        plt.rcParams['figure.subplot.left'] = 0.162
#         plt.rcParams['markeredgewidth']=1
        plt.rcParams['axes.prop_cycle'] = cols_cyc+markers_cyc
        
#    if style=='paperTwoCols':
#        ### For paper - one smaller image full page
#
#        plt.rcParams['figure.figsize'] = [4.2, 2.8]  #Latex text width =0.7*page width = 8.2 inches
#        plt.rcParams['figure.subplot.wspace'] = 0.33
#        plt.rcParams['figure.subplot.top'] = 0.87
#        plt.rcParams['figure.subplot.bottom'] = 0.16
#        plt.rcParams['figure.subplot.right'] = 0.97
#        plt.rcParams['figure.subplot.left'] = 0.15
#        plt.rcParams['axes.prop_cycle'] = cols_cyc

    elif style=='paperFull':
        ### For paper - full page
        plt.rcParams['figure.figsize'] = [6.2, 3.0]  #Latex text width =0.7*page width = 8.2 inches
        # plt.rcParams['figure.figsize'] = [3, 3.4]  #Latex text width =0.7*page width = 8.2 inches
        plt.rcParams['figure.subplot.wspace'] = 0.33
        plt.rcParams['figure.subplot.top'] = 0.87
        plt.rcParams['figure.subplot.bottom'] = 0.14
        plt.rcParams['figure.subplot.right'] = 0.97
        plt.rcParams['figure.subplot.left'] = 0.10
        plt.rcParams['axes.prop_cycle'] = cols_cyc

    elif style=="paperThreeRows":
        plt.rcParams['figure.figsize'] = [3.0, 5.3]  
        plt.rcParams['figure.subplot.hspace'] = 0.20
        plt.rcParams['figure.subplot.top'] = 0.99
        plt.rcParams['figure.subplot.bottom'] = 0.08
        plt.rcParams['figure.subplot.right'] = 0.99
        plt.rcParams['figure.subplot.left'] = 0.195
        plt.rcParams['axes.prop_cycle'] = cols_cyc
    elif style=="paperTwoRows":
        plt.rcParams['figure.figsize'] = [3.0, 3.9] 
        plt.rcParams['figure.subplot.hspace'] = 0.43
        plt.rcParams['figure.subplot.top'] = 0.91
        plt.rcParams['figure.subplot.bottom'] = 0.11
        plt.rcParams['figure.subplot.right'] = 0.99
        plt.rcParams['figure.subplot.left'] = 0.19
        plt.rcParams['axes.prop_cycle'] = cols_cyc
    else:
        raise ValueError(f"Did not find the style = {style}. Available: paper, presentation-square, paperFull, paperThreeRows, paperTwoRows.")



