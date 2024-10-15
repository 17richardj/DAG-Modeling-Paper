import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from matplotlib.ticker import MaxNLocator
import os

def read_data(filename):
    """Read the data from an Excel file."""
    data = pd.read_excel(filename)
    x = data['Time']  # Replace 'Time' with the actual column name in your Excel file
    return x, data

def ode_system(y, t, k, j, KaOpt, KpOpt, VsercaOpt, KsercaOpt, k2Opt, ki, trd, tsd, rdf, sdf):
    """Define the ODE system."""
    dydt = np.zeros(8)
    k_it = ki * (0.6 ** (0.5 * t)) * k ** 2 / (0.2 + k ** 2)
    deltaPlc = 0.325 * j ** -0.324 if j != 0 else 1
    k_it *= deltaPlc

    de_ft = 1 - (rdf * np.exp(-trd) / t) + (sdf * np.exp(-tsd / t)) if t != 0 else 1

    dgkalpha = y[5] ** 2 / (2.66 + y[5] ** 2) * 0.01

    dydt[0] = -y[0] * k_it * de_ft * t      # pip2local
    dydt[1] = y[0] * k_it * de_ft * t - y[1] * dgkalpha * t  # dag
    dydt[2] = y[1] * kii * t - y[2] * kiii * t  # pa
    dydt[3] = y[0] * k_it * de_ft * t - y[3] * kv * t  # ip3
    dydt[4] = y[2] * kiii * t - y[4] * kiv * t  # pip
    dydt[5] = t * (k1 * ((y[5] / (y[5] + KaOpt)) ** 3) * ((y[3] / (y[3] + KpOpt)) ** 3) + k2Opt * (100 - y[5])) - VsercaOpt * (y[5] ** 2 / (KsercaOpt + y[5]) ** 2) * t
    dydt[6] = y[4] * kiv - y[6] * k_it * de_ft * t
    dydt[7] = y[0] + y[6]

    # Prevent values from becoming too large
    max_value = 1e5
    dydt = np.clip(dydt, None, max_value)
    return dydt

def solve_ode_system(k, j, KaOpt, KpOpt, VsercaOpt, KsercaOpt, k2Opt, initial_conditions, t):
    """Solve the ODE system using odeint."""
    solution = odeint(ode_system, initial_conditions, t, args=(k, j, KaOpt, KpOpt, VsercaOpt, KsercaOpt, k2Opt, ki, trd, tsd, rdf, sdf))
    return solution

def generate_contour_plot(X, Y, Z):
    """Generate contour plot with zoomed inset."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.colormaps["rainbow"].with_extremes(under="white", over="yellow")

    contour_plot = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.9, levels=np.linspace(0, 100, 20))
    plt.title('PLC', fontsize=30)
    plt.xlabel('DGKI Concentration (µM)', fontsize=30)
    plt.ylabel('IP3 Concentration (µM)', fontsize=30)

    cbar = plt.colorbar(contour_plot, pad=0.05)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.set_label('PLC Activity (% Basal)', fontsize=30)

    # Adding zoomed inset
    sub_axes = ax.inset_axes([0.4, 0.4, 0.5, 0.5])
    sub_contour = sub_axes.contourf(X[1:31, 0:5], Y[1:31, 0:5], Z[1:31, 0:5], cmap='rainbow', levels=20)
    ax.indicate_inset_zoom(sub_axes, edgecolor="black", facecolor="white", alpha=0.3, linewidth=4)
    sub_axes.tick_params(labelsize=20)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

def main():
    """Main function to execute the ODE solution and plot."""
    # Read data from Excel
    filename = 'oagdata.xlsx'
    x, data = read_data(filename)

    # Parameters and initial conditions
    oag = [0, 0.1, 1, 3, 5, 10, 20, 30]
    agonist = np.arange(0, 3.1, 0.1)
    t_start = 0
    t_end = max(x)
    time_step = 0.1
    t = np.arange(t_start, t_end + time_step, time_step)
    shape_3d = (len(agonist), len(oag), len(t))
    empty_3d_array = np.empty(shape_3d)

    Z = np.zeros((len(agonist), len(oag)))  # Initialize Z for contour plot

    for k_index, k in enumerate(agonist):
        for j_index, j in enumerate(oag):
            # Set optimized parameters based on j
            if j == 0:
                KaOpt, KpOpt, VsercaOpt = 0.000001, 1.5, 5.31
            else:
                KaOpt = 0.82 + 0.338 * np.log(j)
                KpOpt = 1.58 - 0.135 * j + 0.00682 * j ** 2 - 0.000106 * j ** 3
                VsercaOpt = 6.4 - 1.51 * j + 0.354 * j ** 2 - 0.0199 * j ** 3 + 0.000331 * j ** 4

            KsercaOpt = 1.7 + 0.393 * j - 0.506 * j ** 2 + 0.126 * j ** 3 if j <= 5 else 382 * np.exp(-0.808 * j)
            k2Opt = 0.0282 * np.exp(-0.762 * j)
            initial_conditions = [10, 0, k, 0, 0.0, 0, 0, 0.01]

            # Solve the ODE system
            solution = solve_ode_system(k, j, KaOpt, KpOpt, VsercaOpt, KsercaOpt, k2Opt, initial_conditions, t)
            Z[k_index, j_index] = np.mean(solution[:, 0])  # Example: Using mean of solution for Z

    # Generate contour plot
    X, Y = np.meshgrid(oag, agonist)
    generate_contour_plot(X, Y, Z)

    # Save the figure to the desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    plt.savefig(os.path.join(desktop_path, 'dgkiIP3PLC.tiff'), format='tiff', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
