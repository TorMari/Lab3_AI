import random
import math
import matplotlib.pyplot as plt
import numpy as np


def generate_seq(N):
   return [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(N)]
     
def euclidean_distance(point1, point2):
   return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def k_means_clustering(data, k, max_iterations=100):
   centroids = random.sample(data, k)
   for _ in range(max_iterations):
      clusters = [[] for _ in range(k)]
      for point in data:
         distances = [euclidean_distance(point, centroid) for centroid in centroids]
         cluster_index = distances.index(min(distances))
         clusters[cluster_index].append(point)
      new_centroids = []
      for cluster in clusters:
         centroid = tuple(sum(coord) / len(cluster) for coord in zip(*cluster))
         new_centroids.append(centroid)
      if new_centroids == centroids:
         break
      centroids = new_centroids
   return clusters, centroids


def fuzzy_c_means(data, k, m, max_iterations=100, epsilon=1e-6):
    n = len(data)
    centers = random.sample(data, k)
    membership_matrix = np.random.rand(n, k)
    for _ in range(max_iterations):
        prev_membership_matrix = membership_matrix.copy()
        distances = np.array([[euclidean_distance(point, center) for center in centers] for point in data])
        distances = np.where(distances == 0, epsilon, distances) 
        membership_matrix = 1 / np.power(distances, 2 / (m - 1))
        membership_matrix = membership_matrix / np.sum(membership_matrix, axis=1, keepdims=True)
        centers = np.dot(membership_matrix.T, data) / np.sum(membership_matrix, axis=0, keepdims=True).T
        if np.allclose(prev_membership_matrix, membership_matrix, atol=epsilon):
            break
    clusters = [[] for _ in range(k)]
    for i in range(n):
        cluster_index = np.argmax(membership_matrix[i])
        clusters[cluster_index].append(data[i])
    return clusters, centers



def visualize_clusters(clusters1, clusters2, center1, center2):
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'yellow', 'pink', 'olive', 'cyan', 'brown'] 
    plt.subplot(1, 2, 1) 
    for i, cluster in enumerate(clusters1):
        x = [point[0] for point in cluster]
        y = [point[1] for point in cluster]
        plt.scatter(x, y, color=colors[i % len(colors)], label=f'Cluster {i+1} ({len(cluster)})')
    for center in center1:
        plt.scatter(center[0], center[1], color='black', marker='x')
    plt.title('K-means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.subplot(1, 2, 2) 
    for i, cluster in enumerate(clusters2):
        x = [point[0] for point in cluster]
        y = [point[1] for point in cluster]
        plt.scatter(x, y, color=colors[i % len(colors)], label=f'Cluster {i+1} ({len(cluster)})')
    for center in center2:
        plt.scatter(center[0], center[1], color='black', marker='x')
    plt.title('Fuzzy C-means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.tight_layout()
    plt.show()


def cluster_size(cluster):
    return len(cluster)

def weighted_average_cluster_size(clusters, distance_measure):
    total_weighted_size = 0
    total_weight = 0
    for cluster in clusters:
        size = cluster_size(cluster)
        weight = distance_measure(cluster)
        total_weighted_size += size * weight
        total_weight += weight
        print(size, end=' ')
    print('')
    return total_weighted_size / total_weight


N = 5000
k = 6
m = 2 
data = generate_seq(N)

k_means_clusters, center1 = k_means_clustering(data, k)
fuzzy_c_means_clusters, center2 = fuzzy_c_means(data, k, m)

k_means_weighted_average_size = weighted_average_cluster_size(k_means_clusters, cluster_size)
fuzzy_c_means_weighted_average_size = weighted_average_cluster_size(fuzzy_c_means_clusters, cluster_size)

print("Оцінка середньо-зваженого розміру для методу k-means: ", k_means_weighted_average_size)
print("Оцінка середньо-зваженого розміру для методу fuzzy c-means: ", fuzzy_c_means_weighted_average_size)

visualize_clusters(k_means_clusters, fuzzy_c_means_clusters, center1, center2)


