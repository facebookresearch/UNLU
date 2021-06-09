# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Plot the figures from script
## Since Jupyter is so kind to lose all my data

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D
from argparse import ArgumentParser

# Change your path to save file here
loc = ""

## Load all results
## Aggregate scores
datas = [
    "mnli_m_dev",
    "mnli_mm_dev",
    "snli_dev",
    "snli_test",
    "anli_r1_dev",
    "anli_r2_dev",
    "anli_r3_dev",
]
# datas = ['ocnli_dev']
models = ["roberta.large.mnli", "bart.large.mnli", "distilbert.mnli"] + [
    "infersent.mnli",
    "convnet.mnli",
    "bilstmproj.mnli",
]
# models = ['infersent.mnli','convnet.mnli','bilstmproj.mnli']
# models = ['chinese.roberta.large.ocnli']
# models = ['distilbert.mnli']
# models = ['infersent.ocnli', 'convnet.ocnli', 'bilstm.ocnli']
rand = "rand_100_p_1.0_k_0.0_stop_False_punct_True"
dfs = []
for model in models:
    for data in datas:
        p = Path(loc) / data / rand / "outputs" / f"{model}.jsonl"
        print(p)
        df_ = pd.read_json(p, lines=True)
        df_["Eval Data"] = data
        dfs.append(df_)

dfs = pd.concat(dfs)

# Load csvs
dfs_csvs = []

for eval_data in datas:
    for model in models:
        dfr = pd.read_csv(
            f"{loc}/{eval_data}/rand_100_p_1.0_k_0.0_stop_False_punct_True/outputs/{model}.csv"
        )
        dfr["eval_data"] = eval_data
        dfr["model_name"] = model
        dfs_csvs.append(dfr)

dfs_csvs = pd.concat(dfs_csvs)


cols = [
    "Model",
    "Eval Data",
    "Original Accuracy",
    "Max Accuracy",
    "orig_correct_cor_mean",
    "flipped_cor_mean",
    "Correct > Random Percentage",
]
dfs_m = dfs[cols].copy()
clean_names = {
    "infersent.mnli": "InferSent",
    "convnet.mnli": "ConvNet",
    "bilstmproj.mnli": "BiLSTM",
    "roberta.large.mnli": "RoBERTa (large)",
    "bart.large.mnli": "BART (large)",
    "distilbert.mnli": "DistilBERT",
    "chinese.roberta.large.ocnli": "RoBERTa-L",
    "infersent.ocnli": "InferSent",
    "convnet.ocnli": "ConvNet",
    "bilstm.ocnli": "BiLSTM",
}
dfs_m["Model"] = dfs_m["Model"].apply(lambda x: clean_names[x])
dfs_csvs["is_correct"] = dfs_csvs.label == dfs_csvs.orig_pred
dfs_csvs["model_name"] = dfs_csvs["model_name"].apply(lambda x: clean_names[x])
# Bump values by a factor of 100
bump = cols[2:]
for b in bump:
    dfs_m[b] = dfs_m[b] * 100

models = [
    "RoBERTa (large)",
    "BART (large)",
    "DistilBERT",
    "InferSent",
    "ConvNet",
    "BiLSTM",
]
eval_data = [
    "mnli_m_dev",
    "mnli_mm_dev",
    "snli_dev",
    "snli_test",
    "anli_r1_dev",
    "anli_r2_dev",
    "anli_r3_dev",
]
## Plotting Table
markers = ["P", "*", "X", ".", "h", "v"]
marker_size = 100
hatches = [".", "/", "\\", "|", "-", "+", "x"]
hatch_density = 10
data_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]
## Figure aesthetics
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.size"] = 10
plt.rcParams["hatch.linewidth"] = 0.5
plt.rcParams["axes.labelsize"] = "x-large"
## Sanity check
assert len(markers) == len(models)
assert len(hatches) == len(eval_data)
assert len(data_colors) == len(eval_data)


def scatterplot(ax, models, x="Original Accuracy", y="Max Accuracy"):
    for i, ed in enumerate(eval_data):
        for j, model in enumerate(models):
            data = dfs_m[(dfs_m["Eval Data"] == ed) & (dfs_m["Model"] == model)]
            xv = data[x]
            yv = data[y]
            ax.scatter(
                xv,
                yv,
                ec=data_colors[i],
                marker=markers[j],
                hatch=hatches[i] * hatch_density,
                facecolors="none",
                s=marker_size,
            )
    return ax


def plot_image_a(ax):
    ax = scatterplot(ax, models)
    ax.set_ylim(20, 110)
    # Create zoomed plot
    ax2 = ax.inset_axes([0.6, 0.13, 0.3, 0.5])
    ax2 = scatterplot(ax2, models[:3])
    ax2.set_xlim(84, 95)
    ax2.set_ylim(98.5, 99.3)
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)
    # ax.set_xlabel("$\mathcal{A}$")
    ax.set_ylabel("$\Omega_{max}$")
    ax.indicate_inset_zoom(ax2)


def plot_image_b(ax):
    ax = scatterplot(ax, models, y=cols[-1])
    ax.set_ylim(20, 110)
    # Create zoomed plot
    ax2 = ax.inset_axes([0.7, 0.13, 0.3, 0.3])
    ax2 = scatterplot(ax2, models[:3], y=cols[-1])
    ax2.set_xlim(87, 91)
    ax2.set_ylim(77, 85)
    # ax.set_xlabel("$\mathcal{A}$")
    ax.set_ylabel("$\Omega_{rand}$")
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)
    ax.indicate_inset_zoom(ax2)


def plot_image_c(ax):
    ax = scatterplot(ax, models, y=cols[4])
    ax.set_ylim(10, 90)
    # Create zoomed plot
    ax2 = ax.inset_axes([0.6, 0.13, 0.3, 0.5])
    ax2 = scatterplot(ax2, models[:3], y=cols[4])
    ax2.set_xlim(85, 93)
    ax2.set_ylim(68, 78)
    ax.set_xlabel("Accuracy ($\mathcal{A}$)")
    ax.set_ylabel("$\mathcal{P}^c$")
    ax.indicate_inset_zoom(ax2)


def plot_image_d(ax):
    ax = scatterplot(ax, models, y=cols[5])
    ax.set_ylim(10, 90)
    # Create zoomed plot
    ax2 = ax.inset_axes([0.73, 0.6, 0.25, 0.35])
    ax2 = scatterplot(ax2, models[:3], y=cols[5])
    ax2.set_xlim(85, 93)
    ax2.set_ylim(35, 41.5)
    ax.set_xlabel("Accuracy ($\mathcal{A})$")
    ax.set_ylabel("$\mathcal{P}^f$")
    ax.indicate_inset_zoom(ax2)


def plot_combined_fig():
    fig, ax = plt.subplots(2, 2, gridspec_kw={"wspace": 0.22, "hspace": 0.05})
    plot_image_a(ax[0][0])
    plot_image_b(ax[0][1])
    plot_image_c(ax[1][0])
    plot_image_d(ax[1][1])

    # Now, plot the extra legend
    model_legends = []
    for mi, model in enumerate(models):
        model_legends.append(
            Line2D(
                [0],
                [0],
                marker=markers[mi],
                color="black",
                label=model,
                markerfacecolor="none",
                markersize=15,
                linewidth=0,
            )
        )

    data_legends = []
    for di, ed in enumerate(eval_data):
        data_legends.append(
            Patch(
                facecolor=data_colors[di], label=ed, hatch=hatches[di] * hatch_density,
            )
        )

    # legend_elements = [
    #     Patch(facecolor="none", edgecolor="b", marker=markers[0], label="Some Model")
    # ]
    legend1 = plt.legend(
        handles=model_legends,
        loc="upper left",
        bbox_to_anchor=(-0.1, -0.25),
        ncol=2,
        labelspacing=1.5,
        frameon=False,
    )

    legend2 = plt.legend(
        handles=data_legends,
        loc="upper right",
        bbox_to_anchor=(-0.1, -0.25),
        ncol=2,
        frameon=False,
    )
    plt.gca().add_artist(legend1)

    plt.savefig("comb_plot_0.png", dpi=300, bbox_inches="tight")
    plt.savefig("comb_plot_all.pdf", dpi=300, bbox_inches="tight")


def plot_entropy():
    plt.figure()
    g = sns.catplot(
        x="eval_data",
        y="prob_entropy",
        hue="is_correct",
        col="model_name",
        kind="box",
        col_wrap=3,
        data=dfs_csvs,
        legend=False,
        sym="",
    )
    g.set_xticklabels(rotation=40)
    g.despine(left=True)
    g.set_titles(row_template="{row_name}", col_template="{col_name}", size=25)
    g.set_ylabels("Average entropy", fontsize=25)
    g.set_xlabels("Dataset", fontsize=25)
    _, xlabels = plt.xticks()
    g.set_xticklabels(xlabels, size=20)
    g.add_legend(fontsize="xx-large")
    plt.savefig("entropy_plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--plot", type=str, default="")
    args = parser.parse_args()

    plot_functions = {"combined": plot_combined_fig, "entropy": plot_entropy}

    if args.plot not in plot_functions:
        raise AssertionError(f"{args.plot} function not defined")
    else:
        plot_functions[args.plot]()
