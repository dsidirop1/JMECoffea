import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from cycler import cycler

#fsize = 15
#tsize = 18tdir = 'in'major = 5.0
#minor = 3.0lwidth = 0.8
#lhandle = 2.0plt.style.use('default')
#plt.rcParams['text.usetex'] = True
#plt.rcParams['font.size'] = fsize
#plt.rcParams['legend.fontsize'] = tsize
#plt.rcParams['xtick.direction'] = tdir
#plt.rcParams['ytick.direction'] = tdir
#plt.rcParams['xtick.major.size'] = major
#plt.rcParams['xtick.minor.size'] = minor
#plt.rcParams['ytick.major.size'] = 5.0
#plt.rcParams['ytick.minor.size'] = 3.0
#plt.rcParams['axes.linewidth'] = lwidth
#plt.rcParams['legend.handlelength'] = lhandle

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

    if style=='paper':
        ### For paper - one smaller image full page

        plt.rcParams['figure.figsize'] = [4.2, 2.8]  #Latex text width =0.7*page width = 8.2 inches
#        plt.rcParams['figure.subplot.wspace'] = 0.33
        plt.rcParams['figure.subplot.top'] = 0.87
        plt.rcParams['figure.subplot.bottom'] = 0.16
        plt.rcParams['figure.subplot.right'] = 0.97
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['axes.prop_cycle'] = cols_cyc

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

#     else:



