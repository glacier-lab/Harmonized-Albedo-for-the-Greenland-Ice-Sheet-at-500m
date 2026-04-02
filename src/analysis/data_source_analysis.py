#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

sns.set_theme(font_scale=1.5, style="darkgrid")

#%%
df_file = "/data_3/shunan_2/AU/hsa500m/hsa500m_preview/daily_scenario_pixel_counts_simplified.csv"
# lookup_file = "/data_3/shunan_2/AU/hsa500m/hsa500m_preview/scenario_code_lookup.csv"
df = pd.read_csv(df_file)
# df_lookup = pd.read_csv(lookup_file)
#%% total pixel analysis
# rename column scenario-1 to CARRA_data_cap, and scenario-2 to CARRA
df.rename(columns={"scenario_-1": "CARRA_data_cap", "scenario_0": "CARRA"}, inplace=True)
# sum all remaining scenario columns to a new column called HSA500m
scenario_cols = [col for col in df.columns if col.startswith("scenario_") and col not in ["CARRA_data_cap", "CARRA"]]
df["HSA500m"] = df[scenario_cols].sum(axis=1)
df["total"] = df["CARRA_data_cap"] + df["CARRA"] + df["HSA500m"]
# calculate fraction of each source
df["date"] = pd.to_datetime(df["date"])

df_plot = df[["date"]].copy()
df_plot["CARRA_data_cap"] = df["CARRA_data_cap"] / df["total"]
df_plot["CARRA"] = df["CARRA"] / df["total"]
df_plot["HSA500m"] = df["HSA500m"] / df["total"]

# print simple stats of the data source counts
print("Data Source Fraction Statistics:")
for source in ["CARRA_data_cap", "CARRA", "HSA500m"]:
    print(f"  {source}: {df[source].sum() / df['total'].sum():.2%}")
# %% plot daily source fractions

fig = plt.figure(figsize=(15, 7), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.4], width_ratios=[1.2, 1.0])
ax_ts = fig.add_subplot(gs[0, :])
ax_pie = fig.add_subplot(gs[1, 0])
ax_bar = fig.add_subplot(gs[1, 1])

colors = {
	"HSA500m": "#08398b",
	"CARRA_data_cap": "#de5239",
	"CARRA": "#188bb4",
}

df_plot = df_plot.sort_values("date")
plot_order = ["CARRA_data_cap", "CARRA", "HSA500m"]
df_plot.set_index("date")[plot_order].plot.area(
	stacked=True,
	ax=ax_ts,
	color=[colors[c] for c in plot_order],
)
if ax_ts.get_legend() is not None:
	ax_ts.get_legend().remove()
ax_ts.set_ylabel("Data Source Fraction")
ax_ts.set_title("a) Daily Data Source Fractions Over Time")
ax_ts.set_xlabel("")
ax_ts.set_xlim(df_plot["date"].min(), df_plot["date"].max())
ax_ts.set_ylim(0, 1)


pie_values = df[plot_order].sum()
pie_colors = [colors[c] for c in plot_order]
ax_pie.pie(
	pie_values,
	labels=plot_order,
	colors=pie_colors,
	autopct="%1.1f%%",
	startangle=90,
	counterclock=False,
	textprops=dict(color="k", )
)
ax_pie.set_title("b) Total Source Fraction")
ax_pie.set_aspect("equal")

# plot data source by season using stacked bar plot
df_plot_season = df_plot[["date", "CARRA_data_cap", "CARRA", "HSA500m"]].copy()
df_plot_season["season"] = df_plot_season["date"].dt.month % 12 // 3 + 1
season_order = [1, 2, 3, 4]
season_labels = ["Winter", "Spring", "Summer", "Autumn"]
df_plot_season["season"] = pd.Categorical(df_plot_season["season"], categories=season_order, ordered=True)
df_plot_season["season_label"] = df_plot_season["season"].cat.rename_categories(season_labels)
df_season = df_plot_season.groupby("season_label", observed=False)[plot_order].sum().reset_index()
# turn to fraction
df_season[plot_order] = df_season[plot_order].div(df_season[plot_order].sum(axis=1), axis=0)
# print seasonal stats
print("\nSeasonal Data Source Fraction Statistics:")
for season in season_labels:
    season_data = df_season[df_season["season_label"] == season]
    print(f"  {season}:")
    for source in plot_order:
        print(f"    {source}: {season_data[source].values[0]:.2%}")

df_season.set_index("season_label")[plot_order].plot.bar(
	stacked=True,
	ax=ax_bar,
	color=[colors[c] for c in plot_order],
)
if ax_bar.get_legend() is not None:
	ax_bar.get_legend().remove()
ax_bar.set_title("c) Seasonal Data Source Fraction")
ax_bar.set_xlabel("")
ax_bar.set_ylabel("Percentage")
ax_bar.set_ylim(0, 1)
ax_bar.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax_bar.tick_params(axis="x", rotation=0)

fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/data_source.pdf", bbox_inches="tight", dpi=300)
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/data_source.png", bbox_inches="tight", dpi=300)
# %%
