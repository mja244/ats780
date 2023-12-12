# Cluster the XAI maps to
# figure out what patterns of
# SST are most important

import numpy as np
from sklearn.cluster import KMeans

xai_maps = np.load('Xtrain_xai.npy')[:,:,:,0] # 3750, 96, 192
print(np.nanmean(xai_maps, axis=(1,2)))

# Reshape to 2D
xai_maps_flat = xai_maps.reshape(xai_maps.shape[0], -1)

# Number of clusters
k = 5

# Apply k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(xai_maps_flat)

## Reshape labels to OG map shape
#clustered_xai = labels.reshape(3750, 96, 192)

# Reshape the cluster centers
cluster_centers_flat = kmeans.cluster_centers_
cluster_centers = cluster_centers_flat.reshape(5, 96, 192)

np.save('cluster_centers', cluster_centers)

