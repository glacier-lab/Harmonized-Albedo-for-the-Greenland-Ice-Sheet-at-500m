#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def setup_plotting_style():
    sns.set_theme(style="darkgrid", font_scale=1.5)

def calculate_metrics(y_true, y_pred):
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_true - y_pred)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r_value ** 2,
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


def create_band_availability_plot(summary):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    plot_summary = summary.sort_values("valid_sr_count", ascending=False)
    sns.barplot(data=plot_summary, x="band", y="valid_sr_count", color="#4C72B0", ax=axes[0])
    axes[0].set_title("Valid SR Samples by Band")
    axes[0].set_xlabel("Band")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    plot_summary = summary.sort_values("valid_pair_count", ascending=False)
    sns.barplot(data=plot_summary, x="band", y="valid_pair_count", color="#DD8452", ax=axes[1])
    axes[1].set_title("Valid Paired Samples with AWS Albedo")
    axes[1].set_xlabel("Band")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig

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

def fit_mlr_and_validate(df, feature_cols, test_size=0.2, random_state=42):
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
    coef_terms = [f"{coef:.6f}*{name}" for coef, name in zip(model.coef_, feature_cols)]
    equation = "albedo_pred = " + f"{model.intercept_:.6f} + " + " + ".join(coef_terms)

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

def create_mlr_train_test_plot(result):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Train panel
    ax = axes[0]
    hb = ax.hexbin(
        result["yhat_train"], result["y_train"],
        gridsize=100, cmap=cmocean.cm.haline, bins="log"
    )
    sns.regplot(x=result["yhat_train"], y=result["y_train"], scatter=False, color="red", ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel("Predicted Albedo")
    ax.set_ylabel("AWS Albedo")
    ax.set_title("Training")
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label(r"$\log_{10}(\mathrm{Count}+1)$")

    m = result["train_metrics"]
    txt = (
        f'N = {m["n"]}\n'
        f'$R^2$ = {m["r_squared"]:.3f}\n'
        f'RMSE = {m["rmse"]:.3f}\n'
        f'MAE = {m["mae"]:.3f}\n'
        f'Bias = {m["bias"]:.3f}\n'
        f'Slope = {m["slope"]:.3f}\n'
        f'Intercept = {m["intercept"]:.3f}'
    )
    ax.text(
        0.05, 0.95, txt, transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    # Test panel
    ax = axes[1]
    hb = ax.hexbin(
        result["yhat_test"], result["y_test"],
        gridsize=100, cmap=cmocean.cm.haline, bins="log"
    )
    sns.regplot(x=result["yhat_test"], y=result["y_test"], scatter=False, color="red", ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel("Predicted Albedo")
    ax.set_ylabel("AWS Albedo")
    ax.set_title("Validation (Testing)")
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label(r"$\log_{10}(\mathrm{Count}+1)$")

    m = result["test_metrics"]
    txt = (
        f'N = {m["n"]}\n'
        f'$R^2$ = {m["r_squared"]:.3f}\n'
        f'RMSE = {m["rmse"]:.3f}\n'
        f'MAE = {m["mae"]:.3f}\n'
        f'Bias = {m["bias"]:.3f}\n'
        f'Slope = {m["slope"]:.3f}\n'
        f'Intercept = {m["intercept"]:.3f}'
    )
    ax.text(
        0.05, 0.95, txt, transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    fig.tight_layout()
    return fig

def main():
    setup_plotting_style()

    aws_path = "/data_3/shunan_2/AU/hsa500m/PROMICE/promice_day.csv"
    sr_path = "/data_3/shunan_2/AU/hsa500m/GCOMC_SR/albedo_gcomc_sr.csv"

    # Choose predictors explicitly if you want:
    feature_cols = ['Rs_VN01', 'Rs_VN02', 'Rs_VN03', 'Rs_VN05', 'Rs_VN07', 
                    'Rs_VN08', 'Rs_VN09', 'Rs_VN10', 'Rs_VN11', 'Rs_SW03']
    # "Rs_VN04", ,"Rs_VN06" are not used due to very low availability after QA masking
    # feature_cols = None

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
    result = fit_mlr_and_validate(df, feature_cols, test_size=0.2, random_state=42)

    print("\nMLR equation:")
    print(result["equation"])
    print("\nTraining metrics:", result["train_metrics"])
    print("Testing metrics:", result["test_metrics"])

    fig_counts = create_band_availability_plot(availability_summary)
    # fig_counts.savefig("/data_3/shunan_2/AU/hsa500m/calibration/print/gcomc_sr_band_availability.png", dpi=300, bbox_inches="tight")

    fig = create_mlr_train_test_plot(result)
    # fig.savefig("/data_3/shunan_2/AU/hsa500m/calibration/print/gcomc_sr_mlr_validation.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()