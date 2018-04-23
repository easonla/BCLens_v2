import numpy as np
import matplotlib.pyplot as plt


def plot_kappa_alpha(Lens):
    """
    plot
    :type Lens: Class.Lens
    """
    title = 'kappa and deflection angle (arcsec)'
    # SkyCoordinate
    alpha = Lens.alpha
    kappa = Lens.kappa
    fig = plt.figure(figsize=(16, 10.4))
    [dyy, dxx] = np.meshgrid(alpha.y / 4.413, alpha.x / 4.413)
    ax1 = plt.axes([0.1, 0.1, 0.7, 0.8])
    cx1 = ax1.contour(dxx, dyy, alpha.data, colors='black', cmap=None, linestyles="solid")
    cax = fig.add_axes([0.95, 0.1, 0.1, 0.8], visible=False)
    plt.colorbar(cx1, cax=cax)
    plt.clabel(cx1, cax=cax, colors="white")
    level = np.linspace(0, 3, 11)
    cx2 = ax1.contourf(dxx, dyy, kappa.data, levels=level)
    for c in cx2.collections:
        c.set_linewidth(0.1)
        c.set_alpha(0.8)
    cax2 = fig.add_axes([0.85, 0.1, 0.02, 0.8])
    plt.colorbar(cx2, cax=cax2)
    x_min = -200
    x_max = 200
    ticksrange = np.linspace(x_min, x_max, 11)
    title = title
    ax1.set_title(title)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([x_min, x_max])
    ax1.set_xticks(ticksrange)
    ax1.set_yticks(ticksrange)
    ax1.set_xlabel("arcsec")
    ax1.set_ylabel("arcsec")
    ax1.set_xlabel("arcsec")
    ax1.set_ylabel("arcsec")
    ax1.grid(True, which="both")
    ax1.legend(loc='upper left')
    return fig
