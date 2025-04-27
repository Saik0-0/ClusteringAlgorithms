import math
import random
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(50)

X = np.concatenate([np.random.normal(loc=-2, scale=1, size=(100, 2)),
                    np.random.normal(loc=3, scale=1, size=(100, 2)),
                    np.random.normal(loc=7, scale=1, size=(100, 2))])


def choose_cluster_for_point(point: list, clusters):
    min_distance = euclidean_distance(point, clusters[0])
    curr_cluster_ind = 0
    for cluster_ind in range(1, len(clusters)):
        distance = euclidean_distance(point, clusters[cluster_ind])
        if distance < min_distance:
            min_distance = distance
            curr_cluster_ind = cluster_ind
    return curr_cluster_ind


def euclidean_distance(point1, point2):
    sum_squared_diff = sum((x - y) ** 2 for x, y in zip(point1, point2))
    return math.sqrt(sum_squared_diff)


def my_kmeans(dataset, amount_of_clusters: int, stop_criterion: float = 0.01):
    h = 0
    clusters = [[float(random.uniform(np.min(X), np.max(X))) for _ in range(len(dataset[0]))]
                for _ in range(amount_of_clusters)]
    clusters = np.array(clusters)

    current_criterion = float('inf')

    list_for_choose_clusters = [clusters.copy()]
    clustered_data = {cluster_key: [] for cluster_key in range(amount_of_clusters)}

    while not current_criterion < stop_criterion:
        # step 1
        clustered_data = {cluster_key: [] for cluster_key in range(amount_of_clusters)}
        for point in dataset:
            cluster_ind = choose_cluster_for_point(point, clusters)
            clustered_data[cluster_ind].append(point)

        # step 2
        prev_clusters = clusters.copy()
        for cluster_ind in range(len(clusters)):
            if clustered_data[cluster_ind]:
                new_coord = (1.0 / len(clustered_data[cluster_ind])) * sum(clustered_data[cluster_ind])
                clusters[cluster_ind] = new_coord
            else:
                clusters[cluster_ind] = dataset[random.randint(0, len(dataset)-1)]
        list_for_choose_clusters.append(clusters)

        # step 3
        h += 1
        current_criterion = sum([euclidean_distance(clusters[k], prev_clusters[k]) for k in range(amount_of_clusters)])

    return clustered_data, clusters


result, cluster_list = my_kmeans(X, 3)


def make_plot(clustering_result: dict, clusters):
    colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'magenta']
    # Создаём фигуру
    plt.figure(figsize=(10, 7))

    # Визуализируем каждый кластер
    for i, points in clustering_result.items():
        points_np = np.array(points)
        plt.scatter(points_np[:, 0], points_np[:, 1], color=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6)

    plt.scatter(clusters[:, 0], clusters[:, 1], c='red', s=50, marker='X', label='Центры кластеров')
    plt.title('Результаты кластеризации методом k-средних')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


make_plot(result, cluster_list)