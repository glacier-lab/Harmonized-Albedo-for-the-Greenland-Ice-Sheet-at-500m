#%%
import pandas as pd
import numpy as np
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean as cmo

sns.set_theme(font_scale=1.5, style="white")

#%%
# -----------------------------------------------------------------------------
# Configuration of harmonization map plotting
# -----------------------------------------------------------------------------
impath_hsa = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff/hsa500m_gapfilled_20210822.tif"
impath_gcomc = "/data_3/shunan_2/AU/hsa500m/GCOMC_SR_albedo/GCOMC_SRalbedo_20210822_500m.tif"
impath_mcd43a3 = "/data_3/shunan_2/AU/hsa500m/MCD43A3_061_bluesky/MCD43A3_BlueskyAlbedo_20210822_500m.tif"
impath_mod10a1 = "/data_3/shunan_2/AU/hsa500m/MOD10A1_cropped/MOD10A1_2021-08-22.tif"
impath_myd10a1 = "/data_3/shunan_2/AU/hsa500m/MYD10A1_cropped/MYD10A1_2021-08-22.tif"
impath_sice = "/data_3/shunan_2/AU/hsa500m/SICE_rebuild/SICE_Albedo_20210822_500m.tif"
impath_vj143ma3 = "/data_3/shunan_2/AU/hsa500m/VIIRS_bluesky/VJ143MA3/VJ143MA3_BlueskyAlbedo_20210822_500m.tif"
impath_vnp43a3 = "/data_3/shunan_2/AU/hsa500m/VIIRS_bluesky/VNP43MA3/VNP43MA3_BlueskyAlbedo_20210822_500m.tif"
impath_carra = "/data_3/shunan_2/AU/hsa500m/CARRA_GL500m_geotiff/CARRA_Albedo_20210822_500m.tif"

# turn impath into a pd dataframe
df_imfiles = pd.DataFrame({
    "dataset": ["MOD10A1", "MYD10A1", "MCD43A3", "VJ143MA3", "VNP43A3", "SICE", "GCOM-C", "CARRA", "HSA500m"],
    "impath": [impath_mod10a1, impath_myd10a1, impath_mcd43a3, impath_vj143ma3, impath_vnp43a3, impath_sice, impath_gcomc, impath_carra, impath_hsa],
    "subplot_label": ["(a) MOD10A1", "(b) MYD10A1", "(c) MCD43A3", "(d) VJ143MA3", "(e) VNP43A3", "(f) SICE", "(g) GCOM-C", "(h) CARRA", "(i) HSA500m"]
})

# Custom color palette  ref: https://gist.github.com/jscarto/6cc7f547bb7d5d9acda51e5c15256b01
BLUE_FLUORITE = [
    '#291b32', '#2a1b34', '#2b1b34', '#2d1c36', '#2f1c38', '#301c39', '#301d3a', '#321d3b', '#331d3d', '#351d3f',
    '#351e40', '#371e41', '#381e43', '#3a1e45', '#3b1f45', '#3c1f46', '#3e1f48', '#3f1f4a', '#401f4c', '#42204d',
    '#43204e', '#44204f', '#462051', '#472052', '#482054', '#4a2056', '#4a2157', '#4c2158', '#4e215a', '#4f215b',
    '#50215d', '#52215e', '#532160', '#552162', '#552263', '#562264', '#582265', '#592267', '#5b2268', '#5c226b',
    '#5e226c', '#5f226e', '#60226f', '#622271', '#632272', '#642274', '#662276', '#672277', '#692278', '#6a227a',
    '#6c227b', '#6e227d', '#6e237e', '#6f247f', '#702480', '#712581', '#722681', '#732683', '#742783', '#752884',
    '#762985', '#772987', '#792a87', '#792b88', '#7a2c89', '#7b2c8a', '#7c2d8a', '#7d2d8c', '#7e2e8d', '#7f2f8d',
    '#80308e', '#813190', '#823191', '#833292', '#843292', '#863393', '#863494', '#873595', '#893596', '#8a3697',
    '#8b3798', '#8b3899', '#8c389a', '#8e399b', '#8e3a9c', '#8f3b9c', '#8f3d9d', '#8f3e9e', '#903f9e', '#90419e',
    '#90439f', '#9044a0', '#9046a0', '#9047a1', '#9049a1', '#914aa2', '#914ca2', '#914ca3', '#914ea3', '#9150a4',
    '#9151a5', '#9153a5', '#9154a6', '#9156a6', '#9157a7', '#9258a7', '#9259a8', '#925aa8', '#925ba9', '#925da9',
    '#925faa', '#9260ab', '#9260ab', '#9263ac', '#9264ac', '#9265ad', '#9266ae', '#9268ae', '#9269ae', '#926aaf',
    '#926bb0', '#926cb0', '#926eb1', '#926fb1', '#9270b2', '#9271b2', '#9273b3', '#9274b3', '#9275b4', '#9277b5',
    '#9277b5', '#9278b6', '#927ab6', '#927bb7', '#927cb7', '#927eb8', '#927fb8', '#9280b9', '#9281ba', '#9282ba',
    '#9284bb', '#9285bb', '#9285bc', '#9187bc', '#9188bd', '#918abd', '#918bbe', '#918cbf', '#918dbf', '#918ec0',
    '#918fc0', '#9191c1', '#9092c2', '#9094c2', '#9094c2', '#9095c3', '#9096c3', '#8f99c4', '#8f9ac5', '#8f9ac5',
    '#8f9bc6', '#8f9cc6', '#8f9dc7', '#8e9fc8', '#8ea0c8', '#8ea2c9', '#8ea3c9', '#8da5ca', '#8da5ca', '#8da6cb',
    '#8da7cb', '#8ca9cc', '#8caacc', '#8caccd', '#8bacce', '#8badce', '#8baecf', '#8ab0d0', '#8ab2d0', '#8ab2d1',
    '#8ab4d1', '#89b4d1', '#89b5d2', '#89b7d2', '#88b8d3', '#88bad4', '#87bad4', '#87bbd5', '#86bdd6', '#86bed6',
    '#86c0d7', '#85c0d7', '#85c1d8', '#84c3d8', '#84c4d9', '#83c5d9', '#83c6da', '#82c8da', '#82c8db', '#81cadc',
    '#81cbdc', '#80ccdd', '#81cddd', '#84cfdd', '#85cfdd', '#87d0dd', '#8ad0de', '#8dd1de', '#8fd2de', '#90d2de',
    '#92d4de', '#95d5de', '#97d5de', '#98d6de', '#9bd7de', '#9dd7df', '#a0d8df', '#a1d9df', '#a2dadf', '#a5dadf',
    '#a7dbdf', '#aadcdf', '#abdddf', '#acdde0', '#afdfe0', '#b1dfe0', '#b3e0e0', '#b4e1e0', '#b7e2e0', '#bae2e1',
    '#bae3e1', '#bee3e2', '#c0e4e3', '#c1e5e3', '#c4e6e3', '#c6e6e4', '#c8e7e4', '#cbe7e5', '#cde8e5', '#cee9e6',
    '#d2e9e7', '#d3eae7', '#d5eae7', '#d8ebe8', '#d9ece8', '#dcece9', '#deedea', '#dfeeea', '#e2eeea', '#e5efeb',
    '#e6f0eb', '#e9f0ec', '#ebf1ed', '#ecf2ed', '#eff3ee', '#f1f3ee'
]
albedo_cmap = colors.ListedColormap(BLUE_FLUORITE)

#%%
# -----------------------------------------------------------------------------
# Load data and plot
# -----------------------------------------------------------------------------
# create a 4x3 subplot to show all 9 datasets and the qa band of HSA500m
fig, axes = plt.subplots(3, 4, figsize=(17, 16))
axes = axes.flatten()
fig.subplots_adjust(left=0.03, right=0.90, top=0.96, bottom=0.04, wspace=0.1, hspace=0.08)

for idx_i, row in enumerate(df_imfiles.itertuples(index=False)):
    dataset = row.dataset
    impath = row.impath

    with rio.open(impath) as src:
        albedo = src.read(1)
        transform = src.transform
        crs = src.crs

    show(albedo, transform=transform, ax=axes[idx_i], cmap=albedo_cmap, vmin=0, vmax=1)
    axes[idx_i].set_title(row.subplot_label, y=1.02, pad=4)
    axes[idx_i].axis("off")

    if dataset == "HSA500m":
        # add a scalebar to the HSA500m map
        scalebar = ScaleBar(
            dx=1.0,
            units="m",
            fixed_value=300,
            fixed_units="km",
            location="lower right",
            frameon=False,
            color="black",
        )
        axes[idx_i].add_artist(scalebar)

        with rio.open(impath) as src:
            qa_band = src.read(2).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                qa_band[qa_band == nodata] = np.nan

            qa_min = float(np.nanmin(qa_band))
            qa_max = float(np.nanmax(qa_band))

            # Plot QA band using its full native value range (not 0-1 scaling).
            imqa_ax = show(
                qa_band,
                transform=transform,
                ax=axes[idx_i+1],
                cmap=getattr(cmo.cm, "thermal"),
                vmin=qa_min,
                vmax=qa_max,
            )
            imqa = imqa_ax.images[-1]
            # Put QA colorbar right next to QA map without affecting subplot layout.
            cax_qa = inset_axes(
                axes[idx_i+1],
                width="4%",
                height="90%",
                loc="center left",
                bbox_to_anchor=(1.03, 0.0, 1, 1),
                bbox_transform=axes[idx_i+1].transAxes,
                borderpad=0,
            )
            cbar = fig.colorbar(imqa, cax=cax_qa, orientation="vertical")
            qa_min_i = int(np.floor(qa_min))
            qa_max_i = int(np.ceil(qa_max))
            if qa_max_i - qa_min_i <= 20:
                qa_ticks = [float(v) for v in range(qa_min_i, qa_max_i + 1)]
                cbar.set_ticks(qa_ticks)
            else:
                cbar.locator = MaxNLocator(integer=True)
                cbar.update_ticks()
            cbar.set_label("QA Band")
            axes[idx_i+1].set_title("(j) HSA500m QA Band", y=1.02, pad=4)
            axes[idx_i+1].axis("off")
            axes[idx_i+2].axis("off")  # hide the last subplot
            axes[idx_i+3].axis("off")  # hide the last subplot


# add colorbar to the right of the figure
sm = plt.cm.ScalarMappable(cmap=albedo_cmap, norm=colors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.05)
cbar.set_label("Albedo")
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/harmonization_maps.png", dpi=300, bbox_inches="tight")
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/harmonization_maps.pdf", dpi=300)
# %%
# -----------------------------------------------------------------------------
# albedo trend mapping
# -----------------------------------------------------------------------------
# impath_trend = "/data_3/shunan_2/AU/hsa500m/trend/hsa500m_trend_monthly_JJA_2000_2024.tif"
# impath_trend = "/data_3/shunan_2/AU/hsa500m/trend/hsa500m_trend_monthly_JJA_2000_2019.tif"
impath_trend = "/data_3/shunan_2/AU/hsa500m/trend/hsa500m_trend_daily_JJA20002024.tif"
with rio.open(impath_trend) as src:
    # Band 1: linear_slope_per_year
    # Band 2: linear_intercept
    # Band 3: linear_pvalue
    # Band 4: mk_tau
    # Band 5: mk_pvalue
    # Band 6: sens_slope_per_year
    linear_slope_per_year = src.read(1)
    linear_intercept = src.read(2)
    linear_pvalue = src.read(3)
    mk_tau = src.read(4)
    mk_pvalue = src.read(5)
    sens_slope_per_year = src.read(6)
    transform = src.transform
    crs = src.crs

# mask out pixels with p-value >= 0.05
linear_slope_per_year[linear_pvalue >= 0.05] = np.nan
mk_tau[mk_pvalue >= 0.05] = np.nan
sens_slope_per_year[mk_pvalue >= 0.05] = np.nan

# plot the trends 1 * 3 subplots for linear slope, mk tau, and sens slope
fig_trend, axes_trend = plt.subplots(1, 3, figsize=(18, 6))
# fig_trend.subplots_adjust(left=0.03, right=0.90, top=0.96, bottom=0.04, wspace=0.1, hspace=0.08)
kwargs = {'format': '%.2f'}

im_ls = show(linear_slope_per_year, transform=transform, ax=axes_trend[0], cmap=getattr(cmo.cm, "balance_r"), vmin=-0.01, vmax=0.01)
sm = plt.cm.ScalarMappable(cmap=getattr(cmo.cm, "balance_r"), norm=colors.Normalize(vmin=-0.01, vmax=0.01))
sm.set_array([])
cbar_ls = fig_trend.colorbar(sm, ax=axes_trend[0], orientation="vertical", **kwargs)
# cbar_ls.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
cbar_ls.set_label("Linear Slope per Year")
# axes_trend[0].set_title("(a) Linear Trend", y=1.02, pad=4)
axes_trend[0].axis("off")

im_mk = show(mk_tau, transform=transform, ax=axes_trend[1], cmap=getattr(cmo.cm, "balance_r"), vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=getattr(cmo.cm, "balance_r"), norm=colors.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar_mk = fig_trend.colorbar(sm, ax=axes_trend[1], orientation="vertical", **kwargs)
# cbar_mk.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
cbar_mk.set_label("Mann-Kendall Tau")
# axes_trend[1].set_title("(b) Mann-Kendall Tau", y=1.02, pad=4)
axes_trend[1].axis("off")

im_sens = show(sens_slope_per_year, transform=transform, ax=axes_trend[2], cmap=getattr(cmo.cm, "balance_r"), vmin=-0.01, vmax=0.01)
sm = plt.cm.ScalarMappable(cmap=getattr(cmo.cm, "balance_r"), norm=colors.Normalize(vmin=-0.01, vmax=0.01))
sm.set_array([])
cbar_sens = fig_trend.colorbar(sm, ax=axes_trend[2], orientation="vertical", **kwargs)
# cbar_sens.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
cbar_sens.set_label("Sen's Slope per Year")
# axes_trend[2].set_title("(c) Sen's Slope", y=1.02, pad=4)
axes_trend[2].axis("off")

# add subplot labels
axes_trend[0].text(0.02, 0.1, "(a)", transform=axes_trend[0].transAxes, va="top", ha="left")
axes_trend[1].text(0.02, 0.1, "(b)", transform=axes_trend[1].transAxes, va="top", ha="left")
axes_trend[2].text(0.02, 0.1, "(c)", transform=axes_trend[2].transAxes, va="top", ha="left")

# %%
