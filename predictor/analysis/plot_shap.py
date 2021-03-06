from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


def plot_shap_values(x, shap_values, val_list, normalizing_dict, value_to_plot, normalize_shap_plots=True):
    """Plots the denormalized values against their shap values, so that
    variations in the input features to the model can be compared to their effect
    on the model. For example plots, see notebooks/08_gt_recurrent_model.ipynb.

    Parameters:
    ----------
    x: np.array
        The input to a model for a single data instance
    shap_values: np.array
        The corresponding shap values (to x)
    val_list: list
        A list of the variable names, for axis labels
    normalizing_dict: dict
        The normalizing dict saved by the `Cleaner`, so that the x array can be
        denormalized
    value_to_plot: str
        The specific input variable to plot. Must be in val_list
    normalize_shap_plots: boolean
        If True, then the scale of the shap plots will be uniform across all
        variable plots (on an instance specific basis).

    A plot of the variable `value_to_plot` against its shap values will be plotted.
    """
    # first, lets isolate the lists
    idx = val_list.index(value_to_plot)

    x_val = x[:, idx].cpu().numpy()

    # we also want to denormalize
    x_val = (x_val * normalizing_dict[value_to_plot]['std']) + \
            normalizing_dict[value_to_plot]['mean']

    shap_val = shap_values[:, idx]

    months = list(range(1, 12))

    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par1.axis["right"].toggle(all=True)

    if normalize_shap_plots:
        par1.set_ylim(shap_values.min(), shap_values.max())

    host.set_xlabel("Months")
    host.set_ylabel(value_to_plot)
    par1.set_ylabel("Shap value")

    p1, = host.plot(months, x_val, label=value_to_plot)
    p2, = par1.plot(months, shap_val, label="shap value")

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    host.legend()

    plt.draw()
    plt.show()
