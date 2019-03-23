from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


def plot_shap_values(x, shap_values, val_list, normalizing_dict, value_to_plot):
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
