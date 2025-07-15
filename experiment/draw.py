"""
This module is used to draw boxplots for the results of the experiment comparing
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

if not os.path.exists("src") or not os.path.exists("experiment"):
    raise RuntimeError("Please run this script from the project root directory.")

cats_data = pd.read_csv("./experiment/experiment_cats_results.csv")
dogs_data = pd.read_csv("./experiment/experiment_dogs_results.csv")
data = pd.concat([cats_data, dogs_data], ignore_index=True)


def save_boxplot(experiment_data, columns, labels, title, ylabel, filename):
    """
    Save a boxplot of the specified columns from the DataFrame.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # With Outliers
    axes[0].boxplot(
        [experiment_data[col] for col in columns],
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "lightblue"},
        showfliers=True,
    )
    axes[0].set_title("With Outliers")
    axes[0].set_ylabel(ylabel)

    # Without Outliers
    axes[1].boxplot(
        [experiment_data[col] for col in columns],
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "lightblue"},
        showfliers=False,
    )
    axes[1].set_title("Without Outliers")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()


save_boxplot(
    experiment_data=data,
    columns=["sliding_time", "gene_time"],
    labels=["Sliding", "GA"],
    title="Time Distribution: Gene vs Sliding (With Outliers)",
    ylabel="Time (seconds)",
    filename="./img/time_boxplot.png",
)

save_boxplot(
    experiment_data=data,
    columns=["pr_dim1", "pr_dim2", "pr_dim3"],
    labels=["Total", "Proportion", "Cutting"],
    title="PR Value Distribution (With Outliers)",
    ylabel="PR Value",
    filename="./img/pr_boxplot.png",
)
