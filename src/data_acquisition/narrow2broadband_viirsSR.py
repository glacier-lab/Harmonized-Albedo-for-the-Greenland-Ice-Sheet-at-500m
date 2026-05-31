"""Build and evaluate a narrow-to-broadband albedo model for VIIRS SR data.

This script pairs PROMICE AWS daily albedo observations with VIIRS SR band
values, filters physically invalid values, evaluates predictor-band availability,
and fits a multiple linear regression model to estimate broadband albedo.

The workflow is sensor-by-sensor:
- VNP09GA
- VJ109GA
- VJ209GA

For each sensor, the script reads the corresponding CSV produced by
extract_from_VIIRS_SR.py, keeps all available band columns in the summary,
and generates a figure with:
1. Band availability for all candidate VIIRS SR bands.
2. Training-set predicted versus observed AWS albedo.
3. Testing-set predicted versus observed AWS albedo.

Saved outputs:
- viirs_sr_n2b_{PRODUCT}.png
- viirs_sr_n2b_{PRODUCT}.pdf
- viirs_sr_n2b_{PRODUCT}_availability.csv
- viirs_sr_n2b_{PRODUCT}_paired_samples.csv

Shunan Feng (shunan.feng@envs.au.dk)
"""

#%%
from typing import cast

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#%%
AWS_PATH = "/data_3/shunan_2/AU/hsa500m/PROMICE/promice_day.csv"
VIIRS_SR_DIR = "/data_3/shunan_2/AU/hsa500m/VIIRS_SR_mosaics"
PRODUCTS = ["VNP09GA", "VJ109GA", "VJ209GA"]
CEN_REINSTALLATION_DATE = "2017-07-25"
CEN_NAME = "CEN"

AWS_EXCLUDE = [
    "FRE", "LYN_L", "LYN_T", "MIT", "NUK_K", "ZAC_A", "ZAC_L", "ZAC_U",
    "CEN", "CP1", "DY2", "EGP", "HUM", "KAN_U", "NAE", "NAU", "NEM",
    "NSE", "SDL", "SDM", "TUN",
]

# Leave out low-availability M-bands in the regression step.
MODEL_EXCLUDE_BANDS = [
    "SurfReflect_M1",
    "SurfReflect_M2",
    "SurfReflect_M3",
    "SurfReflect_M4",
    "SurfReflect_M5",
]


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
    summary = pd.DataFrame(
        {
            "band": feature_cols,
            "valid_sr_count": [df[c].notna().sum() for c in feature_cols],
            "valid_pair_count": [df[[target_col, c]].dropna().shape[0] for c in feature_cols],
        }
    )
    summary["null_sr_count"] = len(df) - summary["valid_sr_count"]
    summary["null_sr_fraction"] = summary["null_sr_count"] / max(len(df), 1)
    summary["all_null"] = summary["valid_sr_count"] == 0
    return summary.sort_values(
        ["valid_pair_count", "valid_sr_count", "band"],
        ascending=[True, True, True],
    )


def load_pair_sr_aws(aws_path, sr_path, feature_cols=None):
    df_aws = pd.read_csv(aws_path)
    df_sr = pd.read_csv(sr_path)

    df_aws["time"] = pd.to_datetime(df_aws["time"])
    df_sr["time"] = pd.to_datetime(df_sr["time"])

    df = pd.merge(df_aws, df_sr, on=["aws", "time"], how="inner")

    if "albedo" not in df.columns:
        raise ValueError("AWS file must contain column: albedo")
    df = df.rename(columns={"albedo": "albedo_aws"})

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {"aws", "time", "albedo_aws"}]

    if len(feature_cols) == 0:
        raise ValueError("No SR feature columns found in the merged table.")

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


def load_sensor_csv(product):
    sr_path = os.path.join(VIIRS_SR_DIR, f"viirs_sr_values_{product}.csv")
    if not os.path.exists(sr_path):
        raise FileNotFoundError(f"Missing input CSV for {product}: {sr_path}")
    return sr_path


def run_sensor(product):
    aws_path = AWS_PATH
    sr_path = load_sensor_csv(product)

    df_sr = pd.read_csv(sr_path, nrows=1)
    candidate_feature_cols = [c for c in df_sr.columns if c not in {"aws", "time"}]

    # Keep all bands in the availability summary.
    df, feature_cols, all_availability_summary, df_merged = load_pair_sr_aws(
        aws_path,
        sr_path,
        feature_cols=candidate_feature_cols,
    )

    availability_path = os.path.join(VIIRS_SR_DIR, f"viirs_sr_n2b_{product}_availability.csv")
    all_availability_summary.to_csv(availability_path, index=False)

    print(f"\n[{product}] Band availability before removing rows with NaNs across all predictors:")
    print(all_availability_summary.to_string(index=False))

    excluded_bands = all_availability_summary.loc[all_availability_summary["all_null"], "band"].tolist()
    if excluded_bands:
        print(f"\n[{product}] Bands with zero valid SR samples after filtering:")
        print(excluded_bands)
    else:
        print(f"\n[{product}] No bands are completely null after filtering.")

    print(
        f"\n[{product}] Rows remaining after requiring AWS albedo and all selected bands to be non-null:",
        len(df),
    )
    print(f"[{product}] Paired samples:", len(df))
    print(f"[{product}] Stations:", df["aws"].nunique())

    df = df[~df["aws"].isin(AWS_EXCLUDE)].copy()

    manual_excluded_bands = [c for c in MODEL_EXCLUDE_BANDS if c in feature_cols]
    if manual_excluded_bands:
        print(f"\n[{product}] Bands manually excluded from regression due to low valid observations:")
        print(manual_excluded_bands)

    excluded_for_model = sorted(set(excluded_bands + manual_excluded_bands))
    usable_feature_cols = [c for c in feature_cols if c not in excluded_for_model]
    if len(usable_feature_cols) == 0:
        raise ValueError(f"[{product}] No usable bands remain after filtering.")

    result = fit_mlr_and_validate(df, usable_feature_cols, test_size=0.3, random_state=42)

    print(f"\n[{product}] MLR equation:")
    print(result["equation"])
    print(f"\n[{product}] Training metrics:", result["train_metrics"])
    print(f"[{product}] Testing metrics:", result["test_metrics"])

    paired_samples_path = os.path.join(VIIRS_SR_DIR, f"viirs_sr_n2b_{product}_paired_samples.csv")
    df.to_csv(paired_samples_path, index=False)

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
        plot_summary["band"].isin(usable_feature_cols),
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
        palette={
            "Selected Bands": "#41414a",
            "Excluded Bands": "#f6bd20",
        },
        ax=ax,
    )
    ax.set_xlabel(f"{product} Bands")
    ax.set_ylabel("Valid Observation Count")
    ax.tick_params(axis="x", rotation=45)
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

    # axes[0].text(0.9, 0.15, "a)", transform=axes[0].transAxes)
    # axes[1].text(0.9, 0.15, "b)", transform=axes[1].transAxes)
    # axes[2].text(0.9, 0.15, "c)", transform=axes[2].transAxes)
    fig.tight_layout()

    png_path = os.path.join(VIIRS_SR_DIR, f"viirs_sr_n2b_{product}.png")
    pdf_path = os.path.join(VIIRS_SR_DIR, f"viirs_sr_n2b_{product}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    # plt.close(fig)

    print(f"[{product}] Saved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def main():
    for product in PRODUCTS:
        run_sensor(product)


if __name__ == "__main__":
    main()
# %%
