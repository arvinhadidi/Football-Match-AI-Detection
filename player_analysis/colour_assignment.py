import cv2 
import matplotlib.pyplot as pyplot
import numpy as np
from sklearn.cluster import KMeans

image_path = "../output_videos/cropped_image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pyplot.imshow(image)
pyplot.show()

top_half_image=  image[0: int(image.shape[0]/2), :]
pyplot.imshow(top_half_image)
pyplot.show()

# Reshape the image into 2d array
image_2d = top_half_image.reshape(-1, 3)

# perform k-means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(image_2d)

# get the cluster labels
labels = kmeans.labels_

# reshape the labels into the orginal image shape
clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

# Display the clustered image
pyplot.imshow(clustered_image)
pyplot.show()

corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
print(non_player_cluster)

player_cluster = 1-non_player_cluster
print(player_cluster)

kmeans.cluster_centers_[player_cluster]