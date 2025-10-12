import numpy as np
import matplotlib.pyplot as plt
from test import read_data_from_txt
def remove_outliers_by_boxplot(points, iqr_threshold=1.5):
    """
    Remove 3D points that are outliers based on box plot analysis of each dimension.

    Parameters:
    points (numpy.ndarray): A Nx3 array of 3D points.
    iqr_threshold (float): The threshold for identifying outliers based on the interquartile range.
                           Default is 1.5, which means points outside 1.5*IQR from the Q3 or Q1 are considered outliers.

    Returns:
    numpy.ndarray: A Mx3 array of filtered 3D points, where M <= N.
    """
    # Extract each dimension
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute the 1st quartile (Q1) and 3rd quartile (Q3)
    Q1_x, Q3_x = np.percentile(x, [25, 75])
    Q1_y, Q3_y = np.percentile(y, [25, 75])
    Q1_z, Q3_z = np.percentile(z, [25, 75])

    # Compute the interquartile range (IQR)
    IQR_x = Q3_x - Q1_x
    IQR_y = Q3_y - Q1_y
    IQR_z = Q3_z - Q1_z

    # Define the lower and upper bounds for non-outlier points
    lower_bound_x, upper_bound_x = Q1_x - iqr_threshold * IQR_x, Q3_x + iqr_threshold * IQR_x
    lower_bound_y, upper_bound_y = Q1_y - iqr_threshold * IQR_y, Q3_y + iqr_threshold * IQR_y
    lower_bound_z, upper_bound_z = Q1_z - iqr_threshold * IQR_z, Q3_z + iqr_threshold * IQR_z

    # Create boolean masks for non-outlier points
    mask_x = (x >= lower_bound_x) & (x <= upper_bound_x)
    mask_y = (y >= lower_bound_y) & (y <= upper_bound_y)
    mask_z = (z >= lower_bound_z) & (z <= upper_bound_z)

    # Combine the masks to get the final mask of non-outlier points
    mask = mask_x & mask_y & mask_z

    # Apply the mask to filter out the outliers
    filtered_points = points[mask]

    return filtered_points

# Example usage:
# Generate some synthetic 3D points (for demonstration purposes)
data = read_data_from_txt('./datasets/3D_sample_experience_data.txt')
points = data  # 100 random points in 3D space

# Add some outliers (points far away from the others)
outliers = np.array([[0, 0, 0], [1, 1, 1], [0.9, 0.95, 0.99], [0.1, 0.15, 0.2]])
points = np.vstack((points, outliers))

# Remove outliers using box plot analysis
filtered_points = remove_outliers_by_boxplot(points)
print(filtered_points)
print("Original number of points:", points.shape[0])
print("Filtered number of points:", filtered_points.shape[0])

