import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import (
    FINAL_DATASET_PATH,
    PLOTS_DIR,
    MODELS_DIR
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


def save_plot(name):

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            name
        )
    )

    plt.close()


def dataset_visualizations(df):

    print(
        "\nGenerating Dataset Visualizations..."
    )

    # --------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------
    plt.figure(figsize=(6, 4))

    # sns.countplot(
    #     x="label",
    #     data=df
    # )
    if "label" in df.columns:

        sns.countplot(
            data=df,
            x="label"
        )

    else:
        print("Skipping label plot")
    
    plt.title(
        "Parkinson vs Healthy"
    )

    save_plot(
        "label_distribution.png"
    )

    # --------------------------------
    # GENDER
    # --------------------------------
    # if "gender" in df.columns:

    #     plt.figure(figsize=(6, 4))

        # sns.countplot(
        #     x="gender",
        #     hue="label",
        #     data=df
        # )
    if (
    "gender" in df.columns and
    "label" in df.columns
            ):

        sns.countplot(
            data=df,
            x="gender",
            hue="label"
        )

        plt.title(
            "Gender vs Label"
        )

        save_plot(
            "gender_vs_label.png"
        )

    # --------------------------------
    # SEASON
    # --------------------------------
    # if "season" in df.columns:

    #     plt.figure(figsize=(6, 4))

    #     sns.countplot(
    #         x="season",
    #         hue="label",
    #         data=df
    #     )
    if (
        "season" in df.columns and
        "label" in df.columns
    ):

        sns.countplot(
            data=df,
            x="season",
            hue="label"
        )
        plt.title(
            "Season vs Label"
        )

        save_plot(
            "season_vs_label.png"
        )

    # --------------------------------
    # HISTOGRAMS
    # --------------------------------

    numeric_features = [

        "age",
        "MMSE",
        "NAART",
        "disease_duration",

        "delta",
        "theta",
        "alpha",
        "beta"

    ]

    for feature in numeric_features:

        if feature in df.columns:

            plt.figure(figsize=(6, 4))

            sns.histplot(
                data=df,
                x=feature,
                hue="label",
                kde=True
            )

            plt.title(
                feature
            )

            save_plot(
                f"{feature}_hist.png"
            )

    # --------------------------------
    # BOXPLOTS
    # --------------------------------

    eeg_features = [

        "delta",
        "theta",
        "alpha",
        "beta",

        "delta_theta_ratio",
        "alpha_theta_ratio",
        "beta_alpha_ratio"

    ]

    for feature in eeg_features:

        if feature in df.columns:

            plt.figure(figsize=(6, 4))

            sns.boxplot(
                x="label",
                y=feature,
                data=df
            )

            plt.title(
                feature
            )

            save_plot(
                f"{feature}_box.png"
            )

    # --------------------------------
    # VIOLIN PLOTS
    # --------------------------------

    for feature in eeg_features:

        if feature in df.columns:

            plt.figure(figsize=(6, 4))

            sns.violinplot(
                x="label",
                y=feature,
                data=df
            )

            plt.title(
                feature
            )

            save_plot(
                f"{feature}_violin.png"
            )

    # --------------------------------
    # CORRELATION HEATMAP
    # --------------------------------

    plt.figure(
        figsize=(14, 10)
    )

    corr = df.corr(
        numeric_only=True
    )

    sns.heatmap(
        corr,
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Heatmap"
    )

    save_plot(
        "correlation_heatmap.png"
    )

    # --------------------------------
    # PAIRPLOT
    # --------------------------------

    pair_cols = [

        "age",
        "MMSE",

        "delta",
        "theta",
        "alpha",
        "beta",

        "label"
    ]

    pair_cols = [
        c
        for c in pair_cols
        if c in df.columns
    ]

    if len(pair_cols) > 3:

        sns.pairplot(
            df[pair_cols],
            hue="label"
        )

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "pairplot.png"
            )
        )

        plt.close()

    # --------------------------------
    # PCA
    # --------------------------------

    features = [

        c for c in df.columns

        if c not in [
            "participant_id",
            "file",
            "notes",
            "season_name",
            "label"
        ]
    ]

    X = (
        df[features]
        .fillna(0)
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X
    )

    pca = PCA(
        n_components=2
    )

    comps = pca.fit_transform(
        X
    )

    pca_df = pd.DataFrame({

        "PC1": comps[:, 0],

        "PC2": comps[:, 1],

        "label":
        df["label"]
    })

    plt.figure(
        figsize=(8, 6)
    )

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="label"
    )

    plt.title(
        "PCA Projection"
    )

    save_plot(
        "pca_2d.png"
    )


def model_visualizations():

    file_path = os.path.join(
        MODELS_DIR,
        "model_comparison.csv"
    )

    if not os.path.exists(
        file_path
    ):
        return

    df = pd.read_csv(
        file_path
    )

    # --------------------------------
    # TRAIN ACC
    # --------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.barplot(
        x="Model",
        y="Train Accuracy",
        data=df
    )

    plt.xticks(
        rotation=30
    )

    plt.title(
        "Training Accuracy"
    )

    save_plot(
        "train_accuracy.png"
    )

    # --------------------------------
    # TEST ACC
    # --------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.barplot(
        x="Model",
        y="Test Accuracy",
        data=df
    )

    plt.xticks(
        rotation=30
    )

    plt.title(
        "Test Accuracy"
    )

    save_plot(
        "test_accuracy.png"
    )

    # --------------------------------
    # GAP
    # --------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.barplot(
        x="Model",
        y="Gap",
        data=df
    )

    plt.xticks(
        rotation=30
    )

    plt.title(
        "Generalization Gap"
    )

    save_plot(
        "generalization_gap.png"
    )

    monte_path = os.path.join(
        MODELS_DIR,
        "monte_carlo_results.csv"
    )

    if os.path.exists(
        monte_path
    ):

        monte = pd.read_csv(
            monte_path
        )

        plt.figure(
            figsize=(8, 5)
        )

        sns.barplot(
            x="Model",
            y="Test Accuracy",
            data=monte
        )

        plt.xticks(
            rotation=30
        )

        plt.title(
            "Monte Carlo Accuracy"
        )

        save_plot(
            "monte_carlo_accuracy.png"
        )


if __name__ == "__main__":

    df = pd.read_csv(FINAL_DATASET_PATH)

    print("\nDATASET COLUMNS:")
    print(df.columns.tolist())

    # --------------------------
    # FIX LABEL COLUMN
    # --------------------------

    if "label" not in df.columns:

        if "label_x" in df.columns:
            df["label"] = df["label_x"]

        elif "label_y" in df.columns:
            df["label"] = df["label_y"]

        else:
            raise ValueError(
                f"No label column found.\nColumns = {df.columns.tolist()}"
            )

    dataset_visualizations(df)

    model_visualizations()

    print("\nVisualization Complete")
    print(f"Plots saved in: {PLOTS_DIR}")