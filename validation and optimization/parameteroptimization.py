import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.integrate import odeint
from brokenaxes import brokenaxes


def main():
    # Read data from Excel file
    filename = 'oagdata.xlsx'
    data = pd.read_excel(filename)

    # Extract columns
    x = data['Time']  # Replace 'X' with the actual column name in your Excel file

    # Set initial conditions and time array
    initial_conditions = [30, 0, 0, 0, 0.0, 0, 0, 0.01]

    oag = [0, 1, 5, 10, 25, 50, 100]

    max_values = []  # List to store max values for each concentration
    max_model_values = []

    for j in oag:
        # Extract columns for the current concentration
        y1, y2, y3 = get_data_for_concentration(data, j)

        # Calculate the average and standard error of the mean (SEM)
        y = np.mean([y1, y2, y3], axis=0)
        sem = np.std([y1, y2, y3], axis=0) / np.sqrt(3)  # Assuming each data column is an independent sample

        # Set up time array for the ODE integration
        t_start = 0
        t_end = max(x)  # Set the total duration for the ODE integration
        time_step = 0.1  # Set the time step
        t = np.arange(t_start, t_end + time_step, time_step)

        # Interpolate the observed data
        y = np.interp(t, x, y)
        y1 = np.interp(t, x, y1)
        y2 = np.interp(t, x, y2)
        y3 = np.interp(t, x, y3)
        sem = np.interp(t, x, sem)

        # Set up ODE parameters based on concentration
        k1, KaOpt, KpOpt, VsercaOpt, KsercaOpt, k2Opt = setup_ode_parameters(j)

        # Define the ODE system
        def ode_system(y, t):
            dydt = np.zeros(8)
            k_it = ki * (0.6 ** (0.5 * t)) * y[5] ** 2 / (0.2 + y[5] ** 2)

            if t == 0:
                de_ft = 1
            else:
                de_ft = 1 - (rdf * np.exp(-trd) / t) + (sdf * np.exp(-tsd / t))

            dgkalpha = y[5] ** 2 / (10 + y[5] ** 2) * 0.01

            dydt[0] = -y[0] * k_it * de_ft * t  # pip2local
            dydt[1] = y[0] * k_it * de_ft * t - y[1] * dgkalpha * t  # dag
            dydt[2] = y[1] * kii * t - y[2] * kiii * t  # pa
            dydt[3] = y[0] * k_it * de_ft * t - y[3] * kv * t  # ip3
            dydt[4] = y[2] * kiii * t - y[4] * kiv * t  # pip

            dydt[5] = t * (k1 * ((y[5] / (y[5] + KaOpt)) ** 3) * ((y[3] / (y[3] + KpOpt)) ** 3) + k2Opt * (
                        100 - y[5])) - VsercaOpt * (y[5] ** 2 / (KsercaOpt + y[5]) ** 2) * t

            dydt[6] = y[4] * kiv - y[6] * k_it * de_ft * t
            dydt[7] = y[0] + y[6]

            # Error checking to prevent values from going too high
            max_value = 1e5  # Set your maximum allowed value
            for i in range(len(dydt)):
                if dydt[i] > max_value:
                    print(f"Variable {i} exceeded the maximum allowed value. Terminating the simulation.")
                    return np.zeros_like(dydt)

            return dydt

        # Solve the ODE system using odeint
        solution = odeint(ode_system, initial_conditions, t)

        # Calculate area under the curve (AUC)
        auc_insilico = np.trapz(np.maximum(solution[:, 5], 0), dx=time_step)
        aucy1 = np.trapz(np.maximum(y1, 0), dx=time_step)
        aucy2 = np.trapz(np.maximum(y2, 0), dx=time_step)
        aucy3 = np.trapz(np.maximum(y3, 0), dx=time_step)

        max_model_values.append(np.max(solution[:, 5]))
        max_values.append(np.mean([np.max(y1), np.max(y2), np.max(y3)]))

        # Print AUC values
        print(aucy1, aucy2, aucy3, auc_insilico)

    # Plot results
    plot_results(max_values, max_model_values)


def get_data_for_concentration(data, concentration):
    """Extract data for the specified concentration."""
    if concentration == 0:
        return data['veh_a'], data['veh_b'], data['veh_c']
    elif concentration == 1:
        return data['one_a'], data['one_b'], data['one_c']
    elif concentration == 5:
        return data['five_a'], data['five_b'], data['five_c']
    elif concentration == 10:
        return data['ten_a'], data['ten_b'], data['ten_c']
    elif concentration == 25:
        return data['twofive_a'], data['twofive_b'], data['twofive_c']
    elif concentration == 50:
        return data['fifty_a'], data['fifty_b'], data['fifty_c']
    elif concentration == 100:
        return data['max_a'], data['max_b'], data['max_c']
    else:
        raise ValueError(f"Unknown concentration: {concentration}")


def setup_ode_parameters(concentration):
    """Setup ODE parameters based on the concentration."""
    # Define parameters based on concentration
    if concentration == 0:
        return 1.6, 0.00000000001, 1.43, 5.38, 0.238, 0.0414
    elif concentration == 50:
        return 1.1, 0.0000249 * np.exp(0.054 * concentration), 0.692 * concentration ** (
            -0.0109), 1.29 * concentration ** (0.0213), 0.247 - 0.0437 * np.log(concentration), 0.0013381
    elif concentration == 100:
        return 1.1, 0.0000249 * np.exp(0.054 * concentration), 0.692 * concentration ** (
            -0.0109), 1.29 * concentration ** (0.0213), 0.0646, 0.00203
    else:
        return (1.1,
                0.0000249 * np.exp(0.054 * concentration),
                0.692 * concentration ** (-0.0109),
                1.29 * concentration ** (0.0213),
                0.247 - 0.0437 * np.log(concentration),
                0.0013)


def plot_results(max_values, max_model_values):
    """Plot the max values and model values."""
    xaxis = np.arange(len(max_values))
    plt.bar(xaxis - 0.2, max_values, 0.4, label='Experimental Max Values')
    plt.bar(xaxis + 0.2, max_model_values, 0.4, label='Model Max Values')
    plt.xlabel('Concentration')
    plt.ylabel('Max Values')
    plt.title('Max Values vs Concentration')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
