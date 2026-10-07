import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Import the data
df = pd.read_csv("medical_examination.csv")


# Add overweight column
df["overweight"] = (
    (df["weight"] / ((df["height"] / 100) ** 2)) > 25
).astype(int)


# Normalize cholesterol and glucose
df["cholesterol"] = df["cholesterol"].apply(
    lambda x: 0 if x == 1 else 1
)

df["gluc"] = df["gluc"].apply(
    lambda x: 0 if x == 1 else 1
)


def draw_cat_plot():
    # Create DataFrame for categorical plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight"
        ]
    )

    # Group data and count values
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # Create categorical plot
    cat_plot = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        data=df_cat,
        kind="bar"
    )

    # Get the matplotlib Figure
    fig = cat_plot.fig

    # Set figure size
    fig.set_size_inches(10, 5)

    # Save and return figure
    fig.savefig("catplot.png")

    return fig


def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Create mask for upper triangle
    mask = np.triu(
        np.ones_like(corr, dtype=bool)
    )

    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw heat map
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    # Save and return figure
    fig.savefig("heatmap.png")

    return fig