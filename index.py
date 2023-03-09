import numpy as np
import cv2
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Load the image
image = cv2.imread('brain_image.jpg', 0)

# Preprocess the image
otsu_threshold = threshold_otsu(image)
binary_image = image > otsu_threshold
binary_image = binary_image.astype('float32')

# Initialize the FCM algorithm parameters
n_clusters = 2
m = 2
max_iter = 100
tolerance = 1e-4

# Flatten the image into a 1D array
data = binary_image.flatten().reshape(-1, 1)

# Initialize the membership matrix
membership = np.random.dirichlet(np.ones(n_clusters), size=data.shape[0])

# Run the FCM algorithm
for i in range(max_iter):
    # Calculate the cluster centers
    centers = np.zeros((n_clusters, 1))
    for j in range(n_clusters):
        centers[j, :] = np.sum((membership[:, j]**m) * data) / np.sum(membership[:, j]**m)

    # Update the membership matrix
    distances = pairwise_distances_argmin_min(data, centers)[1]
    membership_new = np.zeros_like(membership)
    for j in range(n_clusters):
        membership_new[:, j] = 1 / ((distances / distances[:, np.newaxis])**2 * (1 / centers[j, :])**2 + 1)
    membership_new = membership_new / np.sum(membership_new, axis=1)[:, np.newaxis]

    # Check for convergence
    if np.sum(np.abs(membership_new - membership)) < tolerance:
        break

    membership = membership_new

# Apply K-means clustering to the membership matrix
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(membership)

# Extract the tumor pixels
tumor_pixels = np.where(kmeans.labels_ == 1)[0]

# Highlight the tumor pixels in the image
highlighted_image = np.zeros_like(image)
highlighted_image[tumor_pixels] = image[tumor_pixels]

# Save the highlighted image
cv2.imwrite('brain_tumor.jpg', highlighted_image)
