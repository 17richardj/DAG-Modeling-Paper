import numpy as np
import matplotlib.pyplot as plt
import os

def piecewise_func(x):
    """Define the piecewise function."""
    return np.piecewise(x,
                        [x <= 3, x > 3],
                        [lambda x: (1.39 + 0.387 * x - 0.0878 * x ** 2),
                         lambda x: (0.218 + 0.107 * x - 0.00586 * x ** 2 + 0.0000977 * x ** 3)])

def plot_piecewise_function():
    """Plot the piecewise function and save the figure."""
    # Generate some example data
    x_points = np.array([0, 0.1, 1, 3, 5, 10, 20, 30])
    y_points = np.array([1.41E+00, 1.40E+00, 1.68875145, 1.76E+00, 6.21E-01, 8.04E-01, 8.04E-01, 8.04E-01])

    # Generate x values for smooth curve
    x = np.linspace(0, 30, 1000)

    # Calculate y values using the piecewise function
    y = piecewise_func(x)

    # Plot the scatter points
    plt.scatter(x_points, y_points, color='red', alpha=0.4, s=80)

    # Plot the line
    plt.plot(x, y, color='red', linewidth=4)

    # Add text
    plt.text(0.8, 0.5, 'R² = 1', transform=plt.gca().transAxes,
             verticalalignment='center', horizontalalignment='center',
             fontsize=20, fontweight='bold')

    # Add labels and title
    plt.xlabel('DGKi (µM)', fontsize=30, fontweight='bold')
    plt.ylabel('Param Value', fontsize=30, fontweight='bold')
    plt.title('Kp', fontsize=30, fontweight='bold')

    # Customize the plot appearance
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Set font family
    plt.rcParams['font.family'] = 'Arial'

    # Full path to the desktop (macOS)
    #desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

    # Save the figure to the desktop
    #plt.savefig(os.path.join(desktop_path, 'dgkiKp.tiff'), format='tiff', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

def main():
    """Main entry point for the script."""
    plot_piecewise_function()

if __name__ == "__main__":
    main()
