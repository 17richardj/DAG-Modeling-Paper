import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os


# Function to load and process data
def load_data(filename):
    return pd.read_excel(filename)


# Define the ODE system
def ode_system(y, t, j, params):
    dydt = np.zeros(15)
    ki, rdf, sdf, trd, tsd = params['ki'], params['rdf'], params['sdf'], params['trd'], params['tsd']

    k_it = ki * (0.6 ** (0.5 * t)) * y[5] ** 2 / (0.2 + y[5] ** 2)
    deltaPlc = 1 if j == 0 else 0.325 * j ** -0.324

    de_ft = 1 - (rdf * np.exp(-trd) / t) + (sdf * np.exp(-tsd / t)) if t != 0 else 1
    dgkalpha = y[5] ** 2 / (10 + y[5] ** 2) * 0.01

    # Define ODEs
    dydt[0] = -y[0] * k_it * de_ft * t
    dydt[1] = y[0] * k_it * de_ft * t - y[1] * dgkalpha * 37.5 * 0.01 * t
    dydt[2] = y[1] * params['kii'] * t - y[2] * params['kiii'] * t
    dydt[3] = y[0] * k_it * de_ft * t - y[3] * params['kv'] * t
    dydt[4] = y[2] * params['kiii'] * t - y[4] * params['kiv'] * t
    dydt[5] = t * (1.1 * ((y[5] / (y[5] + params['KaOpt'])) ** 3) * ((y[3] / (y[3] + params['KpOpt'])) ** 3)
                   + params['k2Opt'] * (100 - y[5])) - params['VsercaOpt'] * (
                          y[5] ** 2 / (params['KsercaOpt'] + y[5]) ** 2) * t
    dydt[6] = y[4] * params['kiv'] - y[6] * k_it * de_ft * t
    dydt[7] = y[0] + y[6]

    # Error handling for high values
    max_value = 1e5
    dydt = np.clip(dydt, -max_value, max_value)

    return dydt


# Function to calculate ACF and PACF and plot
def plot_acf_pacf(residuals):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # ACF plot
    plot_acf(residuals, ax=ax[0], color='blue')
    ax[0].set_title('ACF', fontsize=30, fontweight='bold')
    ax[0].tick_params(axis='both', which='major', width=2, labelsize=20)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_linewidth(2)
    ax[0].spines['left'].set_linewidth(2)

    # PACF plot
    plot_pacf(residuals, ax=ax[1], color='red', vlines_kwargs={'colors': 'red'})
    ax[1].set_title('PACF', fontsize=30, fontweight='bold')
    ax[1].tick_params(axis='both', which='major', width=2, labelsize=20)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_linewidth(2)
    ax[1].spines['left'].set_linewidth(2)

    plt.show()


# Main function
def main():
    filename = 'dgki.xlsx'
    data = load_data(filename)

    # Parameters
    params = {
        'ki': 1.3, 'kii': 0.03, 'kiii': 0.004, 'kiv': 0.17, 'kv': 0.01, 'trd': 1.4, 'tsd': 6,
        'rdf': 0.6, 'sdf': 0.43, 'KaOpt': 1.0, 'KpOpt': 1.5, 'VsercaOpt': 5.0, 'KsercaOpt': 1.0, 'k2Opt': 0.1
    }

    oag_concentrations = [0, 1, 5, 10, 30]
    initial_conditions = [10, 0.5, 0, 0, 0.0, 0, 0, 0.01, 0, 0, 0, 0, 0, 10, 0]

    for j in oag_concentrations:
        t = np.arange(0, max(data['Time']), 0.1)

        y1 = data['Veh'] if j == 0 else data[f'{j}uM']
        y = np.mean([y1, y1, y1], axis=0)
        sem = np.std([y1, y1, y1], axis=0) / np.sqrt(3)

        # Interpolate data
        y_interp = np.interp(t, data['Time'], y)

        # Solve ODE
        solution = odeint(ode_system, initial_conditions, t, args=(j, params))

        residuals = y_interp - solution[:, 5]

        # Plot ACF and PACF
        plot_acf_pacf(residuals)

        # Calculate errors
        mae = mean_absolute_error(y_interp, solution[:, 5])
        rmse = mean_squared_error(y_interp, solution[:, 5], squared=False)

        print(f'Concentration {j}uM | MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    # Save plot to desktop
    #desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    #plt.savefig(os.path.join(desktop_path, 'dgkici20.tiff'), format='tiff', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
