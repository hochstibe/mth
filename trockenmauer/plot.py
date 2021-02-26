# FHNW - Institut für Geomatik: Masterthesis 
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 26.02.2021
# plot.py -

import matplotlib.pyplot as plt

from .synth_stone import Stone


def plot_stones(stones: list[Stone]):
    """
    Plot all stones in separate suplots

    :param stones: List of stones
    :return: plot window
    """
    # Caluculate the number of rows / columns of the subplots
    nrows = len(stones) // 4 + 1
    ncols = 4

    fig = plt.figure(figsize=(ncols*3, nrows*2.5))

    for i, s in enumerate(stones):
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
        # axs[i // 3, i % 3] = s.plot()
        s.add_plot_to_ax(ax)

    plt.show()
