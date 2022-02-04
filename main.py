import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpld3
from mpld3 import plugins
np.random.seed(9615)


def fig2html():
    # generate df
    N = 100
    df = pd.DataFrame((.1 * (np.random.random((N, 5)) - .5)).cumsum(0),
                      columns=['a', 'b', 'c', 'd', 'e'],)
    # plot line + confidence interval
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)

    for key, val in df.iteritems():
        l, = ax.plot(val.index, val.values, label=key)
        ax.fill_between(val.index,
                        val.values * .5, val.values * 1.5,
                        color=l.get_color(), alpha=.4)

    # define interactive legend

    handles, labels = ax.get_legend_handles_labels() # return lines and labels
    interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,
                                                             ax.collections),
                                                         labels,
                                                         alpha_unsel=0.5,
                                                         alpha_over=1.5,
                                                         start_visible=True)
    plugins.connect(fig, interactive_legend)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Interactive legend', size=20)

    fig_html = mpld3.fig_to_html(fig)
    text = 'text text'
    return text


if __name__ == "__main__":
    html = fig2html()
