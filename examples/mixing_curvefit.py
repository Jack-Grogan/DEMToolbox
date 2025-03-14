import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DEMToolkit.mixing import lacey_mixing_curve_fit


def plot_fit(x, y, fit_x, fit_y, popt, r2, dimension, save_path):

    fig = plt.figure(figsize=(8, 6), dpi=600, layout='tight')

    plt.scatter(x, y, label="Simulation Data", marker='x', 
                alpha=0.5, color='red')
    plt.plot(fit_x, fit_y, 
             label=r"M = max(1 - (1 - M$_0$)$e^{(-k (t - \tau))}$, M$_0$)", 
             color='black')

    xlim = (0, 14)
    ylim = (-0.01, 1.05)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.fill_between([0,2],[ylim[1], ylim[1]], color="none", hatch="/", 
                     edgecolor="k", linewidth=0.0, alpha=0.5)
    
    plt.fill_between([12,14],[ylim[1], ylim[1]], color="none", hatch="/", 
                     edgecolor="k", linewidth=0.0, alpha=0.5)
    
    plt.annotate("Particle Loading", (1, ylim[1]/2), rotation=90, 
                 ha='center', va='center', fontsize=14, 
                 bbox=dict(facecolor='white',edgecolor='white', alpha=0.5))
    
    plt.annotate("Particle Settling", (13, ylim[1]/2), rotation=90, 
                 ha='center', va='center', fontsize=14, 
                 bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))

    plt.annotate((r"M$_{end}$ = " + f"{y[-1]:.5f}"), xy=(x[-1], y[-1]), 
                 xytext = (15, y[-1]), fontsize=12, va='center', 
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    fit_str = (f"M$_0$ = {fit_y[0]:.5g}    k = {popt[0]:.5g}    "
               f"$\\tau$ = {popt[1]:.5g}    R$^2$ = {r2:.5g}")
    plt.annotate(fit_str, (0.5, -0.15), xycoords='axes fraction', 
                 fontsize=12, ha='center', va='center')

    plt.xlabel("Time (s)")
    plt.ylabel("Lacey Mixing Index M (-)")

    # For matplotlib 3.6.0 and above use ncols instead of ncol
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, 
               fontsize=12, frameon=False)
    
    plt.annotate(f"{dimension} Lacey Mixing", (0.5, 1.15), 
                 xycoords='axes fraction', fontsize=14, ha='center', 
                 va='center')
    
    plt.grid()
    plt.minorticks_on()
    plt.savefig(save_path)
    plt.close(fig)

    return

lacey_df_path = os.path.join(os.path.dirname(__file__),
                             "mixing_analysis", 
                             "lacey_results.csv")
if not os.path.exists(lacey_df_path):
    raise FileNotFoundError(f"File not found: {lacey_df_path}. "
                            "Run mixing.py first.")

save_dir = os.path.join(os.path.dirname(__file__), 
                        "mixing_rate_analysis")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

lacey_df = pd.read_csv(lacey_df_path)

x_lacey = lacey_df["x_lacey"].values
y_lacey = lacey_df["y_lacey"].values
z_lacey = lacey_df["z_lacey"].values
r_lacey = lacey_df["r_lacey"].values
time = lacey_df["time"].values

t0 = 2
tend = 12

fit_res_x = lacey_mixing_curve_fit(time, x_lacey, t0, tend)
fit_res_y = lacey_mixing_curve_fit(time, y_lacey, t0, tend)
fit_res_z = lacey_mixing_curve_fit(time, z_lacey, t0, tend)
fit_res_r = lacey_mixing_curve_fit(time, r_lacey, t0, tend)

x_save_path = os.path.join(save_dir, "lacey_fit_x.png")
y_save_path = os.path.join(save_dir, "lacey_fit_y.png")
z_save_path = os.path.join(save_dir, "lacey_fit_z.png")
r_save_path = os.path.join(save_dir, "lacey_fit_r.png")

plot_fit(time, x_lacey, fit_res_x[2], fit_res_x[4], fit_res_x[0],
         fit_res_x[1], "x", x_save_path)
plot_fit(time, y_lacey, fit_res_y[2], fit_res_y[4], fit_res_y[0],
         fit_res_y[1], "y", y_save_path)
plot_fit(time, z_lacey, fit_res_z[2], fit_res_z[4], fit_res_z[0],
         fit_res_z[1], "z", z_save_path)
plot_fit(time, r_lacey, fit_res_r[2], fit_res_r[4], fit_res_r[0],
         fit_res_r[1], "r", r_save_path)

df = pd.DataFrame({"dimension": ["x", "y", "z", "r"],
                   "k": [fit_res_x[0][0], fit_res_y[0][0], 
                         fit_res_z[0][0], fit_res_r[0][0]],
                   "tau": [fit_res_x[0][1], fit_res_y[0][1], 
                           fit_res_z[0][1], fit_res_r[0][1]],
                   "R2": [fit_res_x[1], fit_res_y[1], 
                          fit_res_z[1], fit_res_r[1]],
                    "t0": [fit_res_x[2][0], fit_res_y[2][0],
                           fit_res_z[2][0], fit_res_r[2][0]],
                    "M0": [fit_res_x[3][0], fit_res_y[3][0],
                            fit_res_z[3][0], fit_res_r[3][0]],
                    "Mend": [x_lacey[-1], y_lacey[-1], 
                             z_lacey[-1], r_lacey[-1]]})

df.to_csv(os.path.join(save_dir, "lacey_fit_results.csv"), index=False)