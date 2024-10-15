import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import os


# Function to define the ODE system
def ode_system(y, t, k1, k2Opt, KaOpt, KpOpt, VsercaOpt, KsercaOpt, j, ki, kiii, kiv, kv):
    """
    Defines the system of ODEs for each condition.
    """
    dydt = np.zeros(8)
    deltaPlc = 1 if j == 0 else 0.325 * j ** -0.324
    k_it = ki * (0.6 ** (0.5 * t))
    de_ft = 1 if t == 0 else 1 - (0.6 * np.exp(-1.4) / t) + (0.43 * np.exp(-6 / t))

    # Define each equation of the system
    dydt[0] = -y[0] * k_it * de_ft * t  # pip2local
    dydt[1] = y[0] * k_it * de_ft * t - y[1] * (y[5] ** 2 / (10 + y[5] ** 2)) * 0.01 * t  # dag
    dydt[2] = y[1] * kiii * t - y[2] * kiii * t  # pa
    dydt[3] = y[0] * k_it * de_ft * t - y[3] * kv * t  # ip3
    dydt[4] = y[2] * kiii * t - y[4] * kiv * t  # pip
    dydt[5] = t * (k1 * ((y[5] / (y[5] + KaOpt)) ** 3) * ((y[3] / (y[3] + KpOpt)) ** 3)
                   + k2Opt * (100 - y[5])) - VsercaOpt * (y[5] ** 2 / (KsercaOpt + y[5]) ** 2) * t  # calcium response
    dydt[6] = y[4] * kiv - y[6] * k_it * de_ft * t  # scavenger
    dydt[7] = y[0] + y[6]

    return dydt


# Function to set up parameters based on OAG concentration
def set_parameters(j):
    """
    Sets the model parameters based on the OAG concentration.
    """
    if j == 0:
        k1, k2Opt = 1.7, 0.0414
        KaOpt, KpOpt, VsercaOpt, KsercaOpt = 1e-11, 1.43, 5.38, 0.238
    elif j == 50:
        k1, k2Opt = 1.1, 0.0013381
        KaOpt = 0.0000249 * np.exp(0.054 * j)
        KpOpt = 0.692 * j ** -0.0109
        VsercaOpt = 1.29 * j ** 0.0213
        KsercaOpt = 0.247 - 0.0437 * np.log(j)
    elif j == 100:
        k1, k2Opt = 1.1, 0.00203
        KaOpt = 0.0000249 * np.exp(0.054 * j)
        KpOpt = 0.692 * j ** -0.0109
        VsercaOpt = 1.29 * j ** 0.0213
        KsercaOpt = 0.0646
    else:
        k1, k2Opt = 1.1, 0.0013
        KaOpt = 0.0000249 * np.exp(0.054 * j)
        KpOpt = 0.692 * j ** -0.0109
        VsercaOpt = 1.29 * j ** 0.0213
        KsercaOpt = 0.247 - 0.0437 * np.log(j)

    # Correct zero values for safety
    KaOpt = max(KaOpt, 1e-11)
    KpOpt = max(KpOpt, 1e-11)
    VsercaOpt = max(VsercaOpt, 1e-11)
    KsercaOpt = max(KsercaOpt, 1e-11)
    k2Opt = max(k2Opt, 1e-11)

    return k1, k2Opt, KaOpt, KpOpt, VsercaOpt, KsercaOpt


# Function to run the simulation and plot results
def run_simulation_and_plot(data, oag_concentrations, column_mappings):
    """
    Runs the ODE simulations for different OAG concentrations and plots the results.
    """
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    count = 0
    colors = ["#7CA0D4", "#A48AD3", "#E995EB", "#BADE86", "#2B8AAE", "#40007F", "#DE757B"]

    for j in oag_concentrations:
        # Extract corresponding data columns for the current OAG concentration
        columns = column_mappings[j]
        y1, y2, y3 = data[columns[0]], data[columns[1]], data[columns[2]]

        # Compute the average and SEM
        y_avg = np.mean([y1, y2, y3], axis=0)
        sem = np.std([y1, y2, y3], axis=0) / np.sqrt(3)

        # Interpolate data to match time points for ODE integration
        time = data['Time']
        t_start, t_end = 0, max(time)
        t = np.arange(t_start, t_end, 0.1)
        y_avg_interp = np.interp(t, time, y_avg)
        sem_interp = np.interp(t, time, sem)

        # Set model parameters for the current OAG concentration
        k1, k2Opt, KaOpt, KpOpt, VsercaOpt, KsercaOpt = set_parameters(j)

        # Initial conditions for the ODE system
        initial_conditions = [30, 0, 0, 0, 0.0, 0, 0, 0.01]

        # Solve the ODE system
        solution = odeint(ode_system, initial_conditions, t, args=(k1, k2Opt, KaOpt, KpOpt, VsercaOpt, KsercaOpt, j, 1.0, 1.0, 1.0, 0.01))

        # Plot the solution
        plt.plot(t, solution[:, 5], color=colors[count], linewidth=6, label=f'OAG {j} µM')
        count += 1

    # Customize plot appearance
    plt.xlabel("Time (s)", fontname="Arial", weight="bold", fontsize=40)
    plt.ylabel("Ca2+ Response", fontname="Arial", weight="bold", fontsize=40)
    plt.tick_params(labelsize=40, length=10, width=4)
    plt.legend(fontsize=20)

    # Customize plot spines
    ax = plt.gca()
    ax.spines['top'].set_color('white')
    ax.spines['top'].set_linewidth(3)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_color('white')
    ax.spines['right'].set_linewidth(3)

    # Show the plot
    plt.show()


def main():
    # Read the data from the Excel file
    filename = 'oagdata.xlsx'
    data = pd.read_excel(filename)

    # Define OAG concentrations and associated column mappings
    oag_concentrations = [0, 1, 5, 10, 25, 50, 100]
    column_mappings = {
        0: ['veh_a', 'veh_b', 'veh_c'],
        1: ['one_a', 'one_b', 'one_c'],
        5: ['five_a', 'five_b', 'five_c'],
        10: ['ten_a', 'ten_b', 'ten_c'],
        25: ['twofive_a', 'twofive_b', 'twofive_c'],
        50: ['fifty_a', 'fifty_b', 'fifty_c'],
        100: ['max_a', 'max_b', 'max_c']
    }

    # Run simulation and plot results
    run_simulation_and_plot(data, oag_concentrations, column_mappings)


if __name__ == "__main__":
    main()
