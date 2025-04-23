import math
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque

np.random.seed(50)

X = np.concatenate([np.random.normal(loc=-2, scale=1, size=(100, 2)),
                    np.random.normal(loc=3, scale=1, size=(100, 2)),
                    np.random.normal(loc=7, scale=1, size=(100, 2))])


def euclidean_distance(point1, point2):
    sum_squared_diff = sum((x - y) ** 2 for x, y in zip(point1, point2))
    return math.sqrt(sum_squared_diff)


def find_nearest_cluster(top_cluster, all_start_clusters):
    min_distance = float('inf')
    nearest_cluster = None
    for cluster in all_start_clusters:
        if cluster is top_cluster:
            continue
        distance = euclidean_distance(top_cluster['center'], cluster['center'])
        if distance < min_distance:
            min_distance = distance
            nearest_cluster = cluster
    return nearest_cluster


def nnc_algorithm(dataset, amount_of_nodes):
    start_clusters = [{'points': [point.tolist()], 'center': point.tolist()} for point in dataset]
    current_clusters = deque()
    while len(start_clusters) > amount_of_nodes:
        if not current_clusters:
            random_first_cluster = random.choice(start_clusters)
            current_clusters.append(random_first_cluster)
            continue
        top_cluster = current_clusters[-1]
        nearest_cluster = find_nearest_cluster(top_cluster, start_clusters)
        if nearest_cluster in current_clusters:
            nearest_position = list(current_clusters).index(nearest_cluster)
            if nearest_position == len(list(current_clusters)) - 2:
                current_clusters.pop()
                current_clusters.pop()

                merged_cluster_points = top_cluster['points'] + nearest_cluster['points']
                merged_cluster_center = np.median(merged_cluster_points, axis=0)
                merged_cluster = {'points': merged_cluster_points, 'center': merged_cluster_center}

                start_clusters.remove(top_cluster)
                start_clusters.remove(nearest_cluster)
                start_clusters.append(merged_cluster)
        else:
            current_clusters.append(nearest_cluster)
    return start_clusters


def make_plot(clustering_result):
    colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'pink']

    plt.figure(figsize=(10, 7))

    # Визуализируем точки каждого кластера
    for i, cluster in enumerate(clustering_result):
        points = np.array(cluster['points'])
        plt.scatter(points[:, 0], points[:, 1],
                    color=colors[i % len(colors)],
                    label=f'Cluster {i + 1}',
                    alpha=0.6)

        # Отмечаем центр кластера
        plt.scatter(cluster['center'][0], cluster['center'][1],
                    c='red',
                    s=150,
                    marker='X',
                    edgecolor='black',
                    linewidth=1)

    plt.title('Результаты кластеризации методом ближайшего соседа')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


clustering = nnc_algorithm(X, 3)
make_plot(clustering)