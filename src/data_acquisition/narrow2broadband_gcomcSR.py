"""Build and evaluate a narrow-to-broadband albedo model for GCOM-C SR data.

This script pairs PROMICE AWS daily albedo observations with GCOM-C surface
reflectance bands, filters physically invalid values, evaluates predictor-band
availability after QA masking, and fits a multiple linear regression model to
estimate broadband albedo.
Note that only bands with native 250m resolution are included. 

The workflow produces a single summary figure containing:
1. The number of valid GCOM-C SR samples available for each candidate band.
2. Training-set predicted versus observed AWS albedo.
3. Testing-set predicted versus observed AWS albedo.

Saved outputs:
- gcomc_sr_n2b.png
- gcomc_sr_n2b.pdf

Shunan Feng (shunan.feng@envs.au.dk)
"""

#%%
from typing import cast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#%%
def calculate_metrics(y_true, y_pred):
    slope, intercept, r_value, p_value, _ = cast(
        tuple[float, float, float, float, float],
        stats.linregress(y_pred, y_true),
    )
    r_squared = r_value ** 2
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_true - y_pred)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r_squared,
        "p_value": p_value,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "n": len(y_true),
    }


def summarize_band_availability(df, feature_cols, target_col="albedo_aws"):
    summary = pd.DataFrame({
        "band": feature_cols,
        "valid_sr_count": [df[c].notna().sum() for c in feature_cols],
        "valid_pair_count": [df[[target_col, c]].dropna().shape[0] for c in feature_cols],
    })
    summary["null_sr_count"] = len(df) - summary["valid_sr_count"]
    summary["null_sr_fraction"] = summary["null_sr_count"] / len(df)
    summary["all_null"] = summary["valid_sr_count"] == 0
    return summary.sort_values(["valid_pair_count", "valid_sr_count", "band"], ascending=[True, True, True])

def load_pair_sr_aws(aws_path, sr_path, feature_cols=None):
    df_aws = pd.read_csv(aws_path)
    df_sr = pd.read_csv(sr_path)

    df_aws["time"] = pd.to_datetime(df_aws["time"])
    df_sr["time"] = pd.to_datetime(df_sr["time"])

    # Merge on paired aws/time
    df = pd.merge(df_aws, df_sr, on=["aws", "time"], how="inner")

    # Target
    if "albedo" not in df.columns:
        raise ValueError("AWS file must contain column: albedo")
    df = df.rename(columns={"albedo": "albedo_aws"})

    # Auto-detect SR predictors if not provided
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("Rs_")]

    if len(feature_cols) == 0:
        raise ValueError("No SR feature columns found. Expected columns like Rs_VN01, Rs_SW01, etc.")

    # Physical range filtering
    df.loc[(df["albedo_aws"] <= 0) | (df["albedo_aws"] >= 1), "albedo_aws"] = np.nan
    for c in feature_cols:
        df.loc[(df[c] < 0) | (df[c] > 1), c] = np.nan

    availability_summary = summarize_band_availability(df, feature_cols, target_col="albedo_aws")

    df_filtered = df.dropna(subset=["albedo_aws"] + feature_cols).copy()
    return df_filtered, feature_cols, availability_summary, df

def fit_mlr_and_validate(df, feature_cols, test_size=0.3, random_state=42):
    X = df[feature_cols].values
    y = df["albedo_aws"].values

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    yhat_train = np.clip(model.predict(X_train), 0, 1)
    yhat_test = np.clip(model.predict(X_test), 0, 1)

    train_metrics = calculate_metrics(y_train, yhat_train)
    test_metrics = calculate_metrics(y_test, yhat_test)

    # Equation text
    coef_terms = [f"{coef:.4f}*{name}" for coef, name in zip(model.coef_, feature_cols)]
    equation = "albedo_pred = " + f"{model.intercept_:.4f} + " + " + ".join(coef_terms)

    out = {
        "model": model,
        "equation": equation,
        "feature_cols": feature_cols,
        "df_train": df_train.copy(),
        "df_test": df_test.copy(),
        "y_train": y_train,
        "y_test": y_test,
        "yhat_train": yhat_train,
        "yhat_test": yhat_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    return out

def main():

    aws_path = "/data_3/shunan_2/AU/hsa500m/PROMICE/promice_day.csv"
    sr_path = "/data_3/shunan_2/AU/hsa500m/GCOMC_SR/albedo_gcomc_sr.csv"
    candidate_feature_cols = ['Rs_VN01', 'Rs_VN02', 'Rs_VN03', "Rs_VN04", 'Rs_VN05', "Rs_VN06",
                              'Rs_VN07', 'Rs_VN08', 'Rs_VN09', 'Rs_VN10', 'Rs_VN11', 'Rs_SW03']
    # evaluate band availability
    df, candidate_feature_cols, all_availability_summary, df_merged = load_pair_sr_aws(
        aws_path,
        sr_path,
        feature_cols=candidate_feature_cols,
    )
    # feature_cols = None
    # Choose predictors explicitly if you want:
    feature_cols = ['Rs_VN01', 'Rs_VN02', 'Rs_VN03', 'Rs_VN05', 'Rs_VN07', 
                    'Rs_VN08', 'Rs_VN09', 'Rs_VN10', 'Rs_VN11', 'Rs_SW03']
    # "Rs_VN04", ,"Rs_VN06" are not used due to very low availability after QA masking
    df, feature_cols, availability_summary, df_merged = load_pair_sr_aws(
        aws_path,
        sr_path,
        feature_cols=feature_cols,
    )
    print("\nBand availability before removing rows with NaNs across all predictors:")
    print(availability_summary.to_string(index=False))

    excluded_bands = availability_summary.loc[availability_summary["all_null"], "band"].tolist()
    if excluded_bands:
        print("\nBands with zero valid SR samples after QA masking:")
        print(excluded_bands)
    else:
        print("\nNo bands are completely null after QA masking.")

    print(
        "\nRows remaining after requiring AWS albedo and all selected bands to be non-null:",
        len(df),
    )
    print("Paired samples:", len(df))
    print("Stations:", df["aws"].nunique())

    # # exclude local glaciers and aws with mean albedo > 0.8
    aws_exclude = ['FRE', 'LYN_L', 'LYN_T', 'MIT', 'NUK_K', 'ZAC_A', 'ZAC_L', 'ZAC_U',
                   'CEN', 'CP1', 'DY2', 'EGP', 'HUM', 'KAN_U', 'NAE', 'NAU', 'NEM', 
                   'NSE', 'SDL', 'SDM', 'TUN']
    df = df[~df["aws"].isin(aws_exclude)].copy()
    result = fit_mlr_and_validate(df, feature_cols, test_size=0.3, random_state=42)

    print("\nMLR equation:")
    print(result["equation"])
    print("\nTraining metrics:", result["train_metrics"])
    print("Testing metrics:", result["test_metrics"])

    sns.set_theme(font_scale=1.5, style="darkgrid")
    haline_cmap = getattr(cmocean.cm, "haline")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 7),
        gridspec_kw={"width_ratios": [1.2, 1.4, 1.4]},
    )

    plot_summary = all_availability_summary.copy()
    plot_summary["selection_status"] = np.where(
        plot_summary["band"].isin(feature_cols),
        "Selected Bands",
        "Excluded Bands",
    )
    plot_summary = plot_summary.sort_values("valid_sr_count", ascending=False)

    ax = axes[0]
    sns.barplot(
        data=plot_summary,
        x="band",
        y="valid_sr_count",
        hue="selection_status",
        dodge=False,
        palette={# pikachu palette
            "Selected Bands": "#41414a",
            "Excluded Bands": "#f6bd20",
        },
        ax=ax,
    )
    ax.set_xlabel("GCOM-C Bands")
    ax.set_ylabel("Valid Observation Count")
    # ax.set_title("Band Availability")
    ax.tick_params(axis="x", rotation=45)
    # move legend to the top of subfigure and show in two columns
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15), title="")

    for ax, yhat_key, y_key, metrics_key, title in [
        (axes[1], "yhat_train", "y_train", "train_metrics", "Training"),
        (axes[2], "yhat_test", "y_test", "test_metrics", "Validation (Testing)"),
    ]:
        hb = ax.hexbin(
            result[yhat_key],
            result[y_key],
            gridsize=100,
            cmap=haline_cmap,
            bins="log",
        )
        sns.regplot(x=result[yhat_key], y=result[y_key], scatter=False, color="red", ax=ax)
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.8)
        ax.set_aspect("equal")
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xlabel("Predicted Albedo")
        ax.set_ylabel("AWS Albedo")
        ax.set_title(title)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label(r"$\log_{10}(\mathrm{Count}+1)$")

        m = result[metrics_key]
        txt = (
            f'N = {m["n"]}\n'
            f'$R^2$ = {m["r_squared"]:.4f}\n'
            f'RMSE = {m["rmse"]:.4f}\n'
            f'MAE = {m["mae"]:.4f}\n'
            f'Bias = {m["bias"]:.4f}\n'
            f'Slope = {m["slope"]:.4f}\n'
            f'Intercept = {m["intercept"]:.4f}'
        )
        ax.text(
            0.05,
            0.95,
            txt,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    # add text annotation for subfigure label
    # axes[0].text(0.9, 0.15,"a)", transform=axes[0].transAxes)
    # axes[1].text(0.9, 0.15,"b)", transform=axes[1].transAxes)
    # axes[2].text(0.9, 0.15,"c)", transform=axes[2].transAxes)
    fig.tight_layout()
    fig.savefig("gcomc_sr_n2b.png", dpi=300, bbox_inches="tight")
    fig.savefig("gcomc_sr_n2b.pdf", dpi=300, bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":
    main()
# %%
