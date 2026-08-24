#%%
import pandas as pd
# %%
df = pd.read_csv(r'C:\Users\au686295\GitHub\postdoc\Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m\stat\calibration_coefficients.csv') 

# %%
print("test data r2 %.3f +/- std %.3f" % (df['test_calib_r_squared'].mean(), df['test_calib_r_squared'].std()))
print("test data rmse %.3f +/- std %.3f" % (df['test_calib_rmse'].mean(), df['test_calib_rmse'].std()))
print("test data bias %.3f +/- std %.3f" % (df['test_calib_bias'].mean(), df['test_calib_bias'].std()))
print("test data mae %.3f +/- std %.3f" % (df['test_calib_mae'].mean(), df['test_calib_mae'].std()))
# %%
