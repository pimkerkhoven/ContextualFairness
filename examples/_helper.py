# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import math

import numpy as np

import matplotlib.pyplot as plt

colors = ["goldenrod", "teal"]


def group_plot(
    groups,
    legend=None,
    legend_title="",
    xticks=None,
    xlabel=None,
    ylabel=None,
    title=None,
    bar_width=0.4,
    fig_width=3.5,
    fig_height=4,
    calc_ylim=False,
    bound=None,
    bound_interval=None,
    show_difference=True,
):
    n_groups = len(groups[0])

    x = np.arange(n_groups)

    end_margin = (bar_width * (len(groups) - 1)) / 2
    margins = np.linspace(-end_margin, end_margin, len(groups))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot bars
    for i, group in enumerate(groups):
        if len(groups) == 1:
            c = colors
        else:
            c = len(group) * [colors[i % len(colors)]]

        ax.bar(
            x + margins[i],
            group,
            bar_width,
            color=c,
        )

    # Set figure labels and titles
    if xticks is not None:
        ax.set_xticks(x, xticks)

    if legend is not None:
        ax.legend(legend, title=legend_title)

    if calc_ylim:
        max_val = 0
        for group in groups:
            for val in group:
                max_val = max(max_val, val)

        ax.set_ylim(0, max_val + 0.2)
    else:
        ax.set_ylim(0, 1)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if bound is not None and bound_interval is not None:
        ax.plot(bound_interval, [bound, bound], color="red", linestyle=(0, (6, 8)))
        ax.text(bound, bound + 0.02, "Bound", ha="center", color="red")
        ax.set_xlim(bound_interval[0], bound_interval[1])

    # Display bar value in each bar
    for i in range(len(groups[0])):
        for j, group in enumerate(groups):
            ax.text(x[i] + margins[j], group[i] / 2, f"{group[i]:.3f}", ha="center")

    # Display differences
    if show_difference:
        for i in range(len(groups[0])):
            max_val = 0
            min_val = math.inf

            for group in groups:
                max_val = max(max_val, group[i])
                min_val = min(min_val, group[i])

            difference = abs(max_val - min_val)

            ax.annotate(
                f"Difference: {difference:.3f}",
                xy=(x[i], max_val + 0.01),
                xytext=(x[i], max_val + 0.01),
                ha="center",
                va="bottom",
                color="black",
            )

    plt.show()


def pprint_result(r):
    for key in r:
        print(key)
        print(f"\tScore: {r[key]['score']}")
        print("\tData:")
        for i, el in zip(r[key]["data"].index, r[key]["data"].to_numpy()):
            print(f"\t {i} \t {el}")
