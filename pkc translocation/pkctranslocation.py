from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
import multiprocessing
import os


def load_image(image_path):
    """Load an image and convert it to a NumPy array."""
    img = Image.open(image_path)
    return np.array(img)


def create_binary_mask(img_array):
    """Create a binary mask from the alpha channel."""
    return img_array[:, :, 3] > 0


def find_edge_points(mask):
    """Find edge points from the binary mask."""
    edge_points = []
    for i in range(1, mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != mask[i - 1, j]:
                edge_points.append((i - 0.5, j + 0.5))

    for i in range(mask.shape[0]):
        for j in range(1, mask.shape[1]):
            if mask[i, j] != mask[i, j - 1]:
                edge_points.append((i + 0.5, j - 0.5))

    return np.array(edge_points, dtype=np.float32)


def generate_random_scatter_points(edge_points, num_molecules):
    """Generate random scatter points within the bounding box of the edge points."""
    min_x, min_y = np.min(edge_points, axis=0)
    max_x, max_y = np.max(edge_points, axis=0)
    rand_points = []

    while len(rand_points) < num_molecules:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        if cv2.pointPolygonTest(edge_points, (x, y), False) >= 0:
            rand_points.append((x, y))

    return np.array(rand_points)


def simulate_molecule_movement(position, num_steps, dt, edge_points):
    """Simulate the movement of a molecule."""
    final_position = position.copy()
    near_edge_positions = []

    for _ in range(num_steps):
        nearest_attraction_point = edge_points[np.argmin(np.linalg.norm(edge_points - final_position, axis=1))]
        direction = nearest_attraction_point - final_position
        direction /= np.linalg.norm(direction)

        # Simulation parameters
        dag = 100
        calcium = 10
        final_position += dt * ((8 + dag) / (20 + dag) / 1.3) * (calcium ** 1.5) / (
                    0.2 + calcium ** 1.2) * direction / np.linalg.norm(direction) ** 2

        if np.any(np.linalg.norm(final_position - edge_points, axis=1) < 0.9):
            near_edge_positions.append(final_position)

    return final_position, near_edge_positions


def run_simulation(molecule_positions, num_steps, dt, edge_points):
    """Run the simulation for all molecules in parallel."""
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(simulate_molecule_movement)(molecule_position, num_steps, dt, edge_points)
        for molecule_position in molecule_positions
    )

    final_positions, near_edge_positions = zip(*results)
    return np.array(final_positions), np.concatenate([positions for positions in near_edge_positions if positions],
                                                     axis=0)


def plot_results(final_positions, near_edge_positions, image_path):
    """Plot the results."""
    img = Image.open(image_path)
    img_greyscale = img.convert('L')
    img_array = np.array(img_greyscale)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_array, cmap='gray_r', vmin=1, alpha=0.35)

    if len(near_edge_positions) > 0:
        plt.scatter(final_positions[:, 1], final_positions[:, 0], color='#107F80', alpha=0.9, label='Edge Points', s=16)
        plt.scatter(near_edge_positions[:, 1], near_edge_positions[:, 0], color='#107F80', edgecolor='black',
                    label='Near Edge Points', s=20)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('off')  # Remove axis
    else:
        print("No points near the edge.")

    # Save the figure
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'nodagcalciumdotscpkc2.tiff')
    plt.savefig(desktop_path, format='tiff', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run the entire process."""
    image_path = 'asm2.tiff'

    img_array = load_image(image_path)
    mask = create_binary_mask(img_array)
    edge_points = find_edge_points(mask)

    num_molecules = 1000
    rand_points = generate_random_scatter_points(edge_points, num_molecules)

    num_steps = 600
    dt = 0.1

    final_positions, near_edge_positions = run_simulation(rand_points, num_steps, dt, edge_points)
    final_positions = final_positions[
        (final_positions[:, 0] >= 0) & (final_positions[:, 0] < img_array.shape[0]) &
        (final_positions[:, 1] >= 0) & (final_positions[:, 1] < img_array.shape[1])
        ]

    plot_results(final_positions, near_edge_positions, image_path)


if __name__ == "__main__":
    main()
