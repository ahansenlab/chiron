"""
written by jacob kaestel hansen
"""
import numpy as np


def generate_gaussian_blob(size, center, sigma=2):
    """
    Generate a 3D Gaussian blob with a specified fractional pixel center.
    Parameters:
    - size (tuple): Size of the 3D volume (depth, height, width).
    - center (tuple): Center of the Gaussian blob (z, y, x) with fractional values.
    - sigma (float): Standard deviation of the Gaussian blob.
    Returns:
    - blob (ndarray): 3D array containing the Gaussian blob.
    """
    z, y, x = np.indices(size)
    z0, y0, x0 = center
    blob = np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))
    return blob

def initial_guesses(ground_truth_centers):
    """
    Generate initial integer guesses for the centers by rounding the ground truth centers.
    Parameters:
    - ground_truth_centers (list of tuples): List of ground truth fractional pixel centers.
    Returns:
    - guesses (list of tuples): List of integer guesses for the blob centers.
    """
    return [(int(round(c[0])), int(round(c[1])), int(round(c[2]))) for c in ground_truth_centers]


def generate_gaussian_blob(size, center, sigma=2):
    """
    Generate a 3D Gaussian blob with a specified fractional pixel center.
    Parameters:
    - size (tuple): Size of the 3D volume (depth, height, width).
    - center (tuple): Center of the Gaussian blob (z, y, x) with fractional values.
    - sigma (float): Standard deviation of the Gaussian blob.
    Returns:
    - blob (ndarray): 3D array containing the Gaussian blob.
    """
    z, y, x = np.indices(size)
    z0, y0, x0 = center
    blob = np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))
    return blob
def generate_ground_truth_set(num_blobs, size, sigma=2):
    """
    Generate a set of Gaussian-like blobs with known fractional pixel centers.
    Parameters:
    - num_blobs (int): Number of blobs to generate.
    - size (tuple): Size of the 3D image volume.
    - sigma (float): Standard deviation of the Gaussian blobs.
    Returns:
    - image (ndarray): 3D image volume with Gaussian blobs.
    - ground_truth_centers (list of tuples): List of ground truth fractional pixel centers.
    """
    image = np.random.normal(0,0.05, size=size)
    ground_truth_centers = []
    for i in range(num_blobs):
        # Generate a random fractional pixel center within the volume bounds
        center = (0,
                  np.random.uniform(20, size[1] - 20),
                  np.random.uniform(20, size[2] - 20))
        #print('center', center)
        ground_truth_centers.append(center)
        # Add the Gaussian blob to the image
        blob = generate_gaussian_blob(size, center, sigma)
        image += blob
    return image, ground_truth_centers

def initial_guesses(ground_truth_centers):
    """
    Generate initial integer guesses for the centers by rounding the ground truth centers.
    Parameters:
    - ground_truth_centers (list of tuples): List of ground truth fractional pixel centers.
    Returns:
    - guesses (list of tuples): List of integer guesses for the blob centers.
    """
    return [(int(round(c[0])), int(round(c[1])), int(round(c[2]))) for c in ground_truth_centers]
