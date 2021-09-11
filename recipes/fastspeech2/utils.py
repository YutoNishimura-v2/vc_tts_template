import matplotlib.pyplot as plt


def plot_mel(data, title):
    fig, axes = plt.subplots(1, 1, squeeze=False)
    axes = axes[0]
    title = None if title is None else title

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    mel, pitch, energy = data
    axes[0].imshow(mel, origin="lower")
    axes[0].set_aspect(2.5, adjustable="box")
    axes[0].set_ylim(0, mel.shape[0])
    axes[0].set_title(title, fontsize="medium")
    axes[0].tick_params(labelsize="x-small", left=False, labelleft=False)
    axes[0].set_anchor("W")

    ax1 = add_axis(fig, axes[0])
    ax1.plot(pitch, color="tomato")
    ax1.set_xlim(0, mel.shape[1])
    ax1.set_ylabel("F0", color="tomato")
    ax1.tick_params(
        labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
    )

    ax2 = add_axis(fig, axes[0])
    ax2.plot(energy, color="darkviolet")
    ax2.set_xlim(0, mel.shape[1])
    ax2.set_ylabel("Energy", color="darkviolet")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(
        labelsize="x-small",
        colors="darkviolet",
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )

    return fig
