from plotapi import Chord
import numpy as np
import os


def main():
    # Set the API key for PlotAPI
    Chord.api_key("#keyhere")

    # Read the CSV file into a NumPy array
    matrix = np.loadtxt('output_matrix.csv', delimiter=',')
    matrix = matrix.tolist()

    # Example matrix data (you can keep this for testing)
    matrix = [
        [100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 50, 0, 50, 0],
        [0, 0, 100, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 100, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 100, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 100, 0, 0, 0, 0],
        [0, 50, 0, 0, 0, 0, 50, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
        [0, 50, 0, 0, 0, 0, 0, 0, 50, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    ]

    # Alternatively, you can use this example matrix for your data
    # matrix = [
    #     ['3.998593363', '0', '1.835036862', '0', '0', '1.833184887', '0', '1.833184887', '0', '0', '0'],
    #     ['0', '6.113867868', '0', '0', '0', '0', '0.065', '0', '1.76', '0.0364', '1.524732132'],
    #     ...
    # ]

    # Create an empty matrix
    empty_matrix = [[0] * 11 for _ in range(11)]

    # Populate the empty matrix
    count = 0
    for i in range(len(empty_matrix)):
        for j in range(len(empty_matrix[i])):
            empty_matrix[i][j] = 1  # Use holddgki10[count] if you want to populate with a specific array
            count += 1

    names = ['Pa', 'PIP2-PLC', 'PIP', 'nPKC', 'cPKC', 'DGK zeta', 'IP3', 'DGK alpha', 'DAG', 'Ca2+', 'PIP2 Recov.']

    # Create and show the Chord diagram
    Chord(
        matrix,
        names,
        colors=["whitesmoke", "black", "whitesmoke", "whitesmoke", "whitesmoke", "whitesmoke", "lightgreen",
                "whitesmoke", "red", "lightgreen", "whitesmoke"],
        pull=[0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        opacity=1,
        colored_diagonals=False,
        curved_labels=True,
        rotate=-50,
        outer_radius_scale=1.15
    ).show()

    # Save the figure to desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    # Uncomment the line below to save the figure as a .tiff file
    # Chord.savefig(os.path.join(desktop_path, 'oag10.tiff'), format='tiff', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
