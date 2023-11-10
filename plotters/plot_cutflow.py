import matplotlib.pyplot as plt
import os
import mplhep as hep
from plotters.pltStyle import pltStyle
pltStyle(style='hep')


def remove_xminor_ticks():
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

figdir = "fig/cutflow"
if not os.path.exists(figdir):
    os.mkdir(figdir)


def plot_cutflow(cutflow, tag_full, ylab, fig_name, title_name=None, alt_tick_labs=None):
    if title_name is None:
        title_name=tag_full
    fig, ax = plt.subplots()
    mc = next(ax._get_lines.prop_cycler)
    cutflow.plot1d(color=mc['color'])
    ax.set_ylabel(ylab)
    # ax.minorticks_off()
    remove_xminor_ticks()
    plt.xticks(rotation=60)
    if alt_tick_labs is not None:
        xticklab = ax.get_xticklabels()
        for lab, lab_txt in zip(xticklab, alt_tick_labs):
            lab.set_text(lab_txt)
    hep.label.exp_text(text=title_name, loc=0)

    fig_name = figdir+"/"+fig_name+'_'+tag_full
    print("Saving plot with the name = ", fig_name)
    plt.savefig(fig_name+'.pdf');
    plt.savefig(fig_name+'.png');
    plt.show()
    