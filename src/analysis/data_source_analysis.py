'''
This script analyzes the data source composition of the HSA500m dataset by comparing it to the CARRA and CARRA with 
data cap. It calculates the fraction of each data source (CARRA data cap, CARRA, HSA_daily, and HSA_16day) for each
day, and then visualizes the results using a time series plot, a pie chart, and a monthly bar plot. 
The script also prints out statistics on the overall and seasonal data source fractions.

Shunan Feng (shunan.feng@envs.au.dk)
'''
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

sns.set_theme(font_scale=1.5, style="darkgrid")

#%%
df_file = "/data_3/shunan_2/AU/hsa500m/hsa500m_preview/daily_scenario_pixel_counts_simplified.csv"
lookup_file = "/data_3/shunan_2/AU/hsa500m/hsa500m_preview/scenario_code_lookup.csv"
calibration_file = "/data_3/shunan_2/AU/hsa500m/calibration/calibration_coefficients.csv"

df = pd.read_csv(df_file)

# ── classify each scenario code as daily or 16-day ──────────────────────────
# The lookup CSV maps code → simplified sensor label.
# The calibration CSV carries scenario_family per scenario_id.
# We replicate the same label-building logic used in preview_hsa500m_gapfilled.py
# to associate each code with a family.

FALLBACK_SENSOR_LABELS = {"mcd43a3", "vj143ma3", "vnp43ma3"}

def _code_family(sensors_str: str) -> str:
    """Return 'daily' if none of the 16-day sensor names appear, else '16day'."""
    parts = {s.strip() for s in str(sensors_str).split("+") if s.strip()}
    return "16day" if parts & FALLBACK_SENSOR_LABELS else "daily"

lookup = pd.read_csv(lookup_file)
lookup["family"] = lookup["sensors"].apply(_code_family)
daily_codes = set(lookup.loc[lookup["family"] == "daily",  "scenario_code"].astype(int))
fallback_codes = set(lookup.loc[lookup["family"] == "16day", "scenario_code"].astype(int))

#%% total pixel analysis
df.rename(columns={"scenario_-1": "CARRA_data_cap", "scenario_0": "CARRA"}, inplace=True)

all_scenario_cols = [c for c in df.columns if c.startswith("scenario_") and c not in ("CARRA_data_cap", "CARRA")]

daily_cols   = [c for c in all_scenario_cols if int(c.split("_", 1)[1]) in daily_codes]
fallback_cols = [c for c in all_scenario_cols if int(c.split("_", 1)[1]) in fallback_codes]

df["HSA_daily"]  = df[daily_cols].sum(axis=1)   if daily_cols   else 0
df["HSA_16day"]  = df[fallback_cols].sum(axis=1) if fallback_cols else 0
df["total"] = df["CARRA_data_cap"] + df["CARRA"] + df["HSA_daily"] + df["HSA_16day"]
df["date"] = pd.to_datetime(df["date"])

df_plot = df[["date"]].copy()
df_plot["CARRA_data_cap"] = df["CARRA_data_cap"] / df["total"]
df_plot["CARRA"]          = df["CARRA"]          / df["total"]
df_plot["HSA_daily"]      = df["HSA_daily"]      / df["total"]
df_plot["HSA_16day"]      = df["HSA_16day"]      / df["total"]

# print simple stats of the data source counts
print("Data Source Fraction Statistics:")
for source in ["CARRA_data_cap", "CARRA", "HSA_daily", "HSA_16day"]:
    print(f"  {source}: {df[source].sum() / df['total'].sum():.2%}")
# %% plot daily source fractions

fig = plt.figure(figsize=(15, 7), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.4], width_ratios=[1.2, 1.0])
ax_ts = fig.add_subplot(gs[0, :])
ax_pie = fig.add_subplot(gs[1, 0])
ax_bar = fig.add_subplot(gs[1, 1])

colors = {
	"HSA_daily":     "#104a62",
	"HSA_16day":     "#419cbd",
	"CARRA_data_cap": "#ffe6a4",
	"CARRA":          "#d55273",
}  # Vaporeon

df_plot = df_plot.sort_values("date")
plot_order = ["CARRA_data_cap", "CARRA", "HSA_16day", "HSA_daily"]
df_plot.set_index("date")[plot_order].plot.area(
	stacked=True,
	ax=ax_ts,
	color=[colors[c] for c in plot_order],
)
# handles, labels = ax_ts.get_legend_handles_labels()
if ax_ts.get_legend() is not None:
	ax_ts.get_legend().remove()
ax_ts.set_ylabel("Percentage")
# ax_ts.set_title("a) Daily Data Source Fractions Over Time")
ax_ts.set_xlabel("")
ax_ts.set_xlim(df_plot["date"].min(), df_plot["date"].max())
ax_ts.set_ylim(0, 1)
ax_ts.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


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
# ax_pie.set_title("b) Total Source Fraction")
ax_pie.set_aspect("equal")
# move pie chart slightly to the left
ax_pie.set_position((-0.05, 0.01, 0.4, 0.4))
ax_pie.legend(
	plot_order,
	loc="center right",
	bbox_to_anchor=(2.55, 0.35),
	frameon=True,
	title="Sources",
)

# plot data source by month using stacked area plot
df_plot_month = df[["date", "CARRA_data_cap", "CARRA", "HSA_daily", "HSA_16day"]].copy()
df_plot_month["month"] = df_plot_month["date"].dt.month
df_month = df_plot_month.groupby("month", observed=False)[plot_order].sum().reset_index()
# turn to fraction	
df_month[plot_order] = df_month[plot_order].div(df_month[plot_order].sum(axis=1), axis=0)
# print monthly stats
print("\nMonthly Data Source Fraction Statistics:")
for month in range(1, 13):
	month_data = df_month[df_month["month"] == month]
	print(f"  Month {month}:")
	for source in plot_order:
		print(f"    {source}: {month_data[source].values[0]:.2%}")
df_month.set_index("month")[plot_order].plot.area(
	stacked=True,
	ax=ax_bar,
	color=[colors[c] for c in plot_order],
)
if ax_bar.get_legend() is not None:
	ax_bar.get_legend().remove()
# ax_bar.set_title("c) Monthly Data Source Fraction")
ax_bar.set_xlabel("")
ax_bar.set_ylabel("Percentage")
ax_bar.set_xticks(range(1, 13))
ax_bar.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax_bar.set_ylim(0, 1)
ax_bar.set_xlim(1, 12)
ax_bar.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


# add text labels to the topleft of each subplot
ax_ts.text(-0.07, 1.03, "a)", transform=ax_ts.transAxes, va="top", ha="right")
ax_pie.text(-0.19, 0.9, "b)", transform=ax_pie.transAxes, va="top", ha="right")
ax_bar.text(-0.13, 1.05, "c)", transform=ax_bar.transAxes, va="top", ha="right")


fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/data_source.pdf", bbox_inches="tight", dpi=300)
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/data_source.png", bbox_inches="tight", dpi=300)
# %%
# # plot data source by season using stacked bar plot
# df_plot_season = df[["date", "CARRA_data_cap", "CARRA", "HSA500m"]].copy()
# df_plot_season["month"] = df_plot_season["date"].dt.month
# # define seasons based on month: 12-2 is DJF, 3-5 is MAM, 7-9 is JJA, 10-12 is SON
# df_plot_season["season"] = df_plot_season["month"].apply(
# 	lambda m: 1 if m in [12, 1, 2] else (2 if m in [3, 4, 5] else (3 if m in [6, 7, 8] else 4))
# )
# season_order = [1, 2, 3, 4]
# season_labels = ["DJF", "MAM", "JJA", "SON"]
# df_plot_season["season"] = pd.Categorical(df_plot_season["season"], categories=season_order, ordered=True)
# df_plot_season["season_label"] = df_plot_season["season"].cat.rename_categories(season_labels)
# df_season = df_plot_season.groupby("season_label", observed=False)[plot_order].sum().reset_index()
# # turn to fraction
# df_season[plot_order] = df_season[plot_order].div(df_season[plot_order].sum(axis=1), axis=0)
# # print seasonal stats
# print("\nSeasonal Data Source Fraction Statistics:")
# for season in season_labels:
#     season_data = df_season[df_season["season_label"] == season]
#     print(f"  {season}:")
#     for source in plot_order:
#         print(f"    {source}: {season_data[source].values[0]:.2%}")

# df_season.set_index("season_label")[plot_order].plot.bar(
# 	stacked=True,
# 	ax=ax_bar,
# 	color=[colors[c] for c in plot_order],
# )
# if ax_bar.get_legend() is not None:
# 	ax_bar.get_legend().remove()
# # ax_bar.set_title("c) Seasonal Data Source Fraction")
# ax_bar.set_xlabel("")
# ax_bar.set_ylabel("Percentage")
# ax_bar.set_ylim(0, 1)
# ax_bar.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
# ax_bar.tick_params(axis="x", rotation=0)