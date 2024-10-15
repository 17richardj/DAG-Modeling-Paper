import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.ticker import MaxNLocator


# Function to define the ODE system
def ode_system(y, t, k1, k2Opt, KaOpt, KpOpt, VsercaOpt, KsercaOpt, deltaPlc, de_ft, ki, t_step):
    """
    Defines the system of ODEs to simulate PLC activity.
    """
    dydt = np.zeros(13)  # Array to hold the rates of change for each variable
    k_it = ki * (0.6 ** (0.5 * t)) * deltaPlc * y[3] ** 2 / (1 + y[3] ** 2)
    dgkalpha = y[5] ** 2 / (2.66 + y[5] ** 2) * 0.01

    dydt[0] = -y[0] * k_it * de_ft * t_step  # pip2local
    dydt[1] = y[0] * k_it * de_ft * t_step - y[1] * y[8] * t_step  # dag
    dydt[2] = y[1] * 0.03 * t_step - y[2] * 0.004 * t_step  # pa
    dydt[3] = y[0] * k_it * de_ft * t_step - y[3] * 0.01 * t_step  # ip3
    dydt[4] = y[2] * 0.004 * t_step - y[4] * 0.17 * t_step  # pip
    dydt[5] = (k1 * ((y[5] / (y[5] + KaOpt)) ** 3) * ((y[3] / (y[3] + KpOpt)) ** 3)
               + k2Opt * (100 - y[5])) * t_step - VsercaOpt * (y[5] ** 2 / (KsercaOpt + y[5]) ** 2) * t_step
    dydt[6] = y[4] * 0.17 - y[6] * k_it * de_ft * t_step  # scavenged variable
    dydt[7] = y[0] + y[6]
    dydt[8] = dgkalpha / t_step  # hypothetical variable
    dydt[9] = ((8 * ((y[5] ** 1.2) / (3.1 + y[5] ** 1.2)) * y[1]) / (20 + y[1])) / t_step
    dydt[10] = y[9] * 0.01 / t_step
    dydt[11] = ((8 * y[1]) / (20 + y[1])) / t_step * 0.1
    dydt[12] = 0  # No changes for the last variable

    # Error checking to prevent values from going too high
    dydt = np.clip(dydt, -1e5, 1e5)

    return dydt


# Function to run the ODE simulation
def run_simulation(ip3, oag, t, initial_conditions, data):
    """
    Runs the simulation for different IP3 and OAG concentrations.
    """
    Z = np.zeros((len(ip3), len(oag)))  # Initialize results array
    for k_index, k in enumerate(ip3):
        for j_index, j in enumerate(oag):
            # Setting the parameters based on OAG concentration
            if j == 0:
                k1, KaOpt, KpOpt, VsercaOpt, KsercaOpt, k2Opt = 1.7, 1e-11, 1.43, 5.38, 0.238, 0.0414
            elif j == 50:
                k1, KaOpt, KpOpt, VsercaOpt, KsercaOpt, k2Opt = 1.1, 2.49e-5 * np.exp(
                    0.054 * j), 0.692 * j ** -0.0109, 1.29 * j ** 0.0213, 0.247 - 0.0437 * np.log(j), 0.0013381
            else:
                k1, KaOpt, KpOpt, VsercaOpt, KsercaOpt, k2Opt = 1.1, 2.49e-5 * np.exp(
                    0.054 * j), 0.692 * j ** -0.0109, 1.29 * j ** 0.0213, 0.0646, 0.00203

            deltaPlc = 0.325 * j ** -0.324 if j != 0 else 1
            de_ft = 1 if t[0] == 0 else 1 - (0.6 * np.exp(-1.4) / t) + (0.43 * np.exp(-6 / t))

            # Solve ODEs
            solution = odeint(ode_system, initial_conditions, t,
                              args=(k1, k2Opt, KaOpt, KpOpt, VsercaOpt, KsercaOpt, deltaPlc, de_ft, 1.3, 0.1))
            Z[k_index, j_index] = np.mean(solution[:, 12])

    return Z


# Function to plot the results
def plot_results(X, Y, Z):
    """
    Creates a contour plot with the simulation results.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    levels = np.linspace(0, 100, 20)
    cmap = plt.get_cmap("rainbow").with_extremes(under="magenta", over="yellow")

    contour_plot = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.9, levels=levels)
    contour_plot.cmap.set_over('red')

    # Add colorbar and adjust formatting
    cbar = plt.colorbar(contour_plot, pad=0.05)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.set_label('PLC Activity (% Basal)', fontsize=30)

    plt.title('PLC Activity', fontsize=30)
    plt.xlabel('OAG Concentration (µM)', fontsize=30)
    plt.ylabel('IP3 Concentration (µM)', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Save the figure
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    plt.savefig(os.path.join(desktop_path, 'oag_IP3_PLC_Activity.tiff'), format='tiff', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Load the data from the Excel file
    filename = 'oagdata.xlsx'
    data = pd.read_excel(filename)

    # Set time parameters
    t_start = 0
    t_end = max(data['Time'])  # Assuming 'Time' is a column in the dataset
    t = np.arange(t_start, t_end + 0.1, 0.1)

    # Define OAG and IP3 concentrations
    oag = [0, 1, 5, 10, 25, 50, 100]
    ip3 = np.arange(0, 3.1, 0.1)

    # Set initial conditions for the ODE system
    initial_conditions = [10, 0, 0, 0, 0.0, 0, 0, 0.01, 0, 0, 0, 0, 0]

    # Run the simulation
    Z = run_simulation(ip3, oag, t, initial_conditions, data)

    # Plot the results
    X, Y = np.meshgrid(oag, ip3)
    plot_results(X, Y, Z)


# Ensure the script runs only if executed as the main program
if __name__ == "__main__":
    main()
