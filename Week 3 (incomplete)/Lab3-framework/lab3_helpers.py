import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.spatial.distance import cdist


def generate_data(num_classes, num_points_per_class, seed):
    np.random.seed(seed)

    clusters = []
    for i in range(num_classes):
        mu_i = np.random.randn(1, 2) * 3
        srd_i = np.random.uniform(1, 2)

        new_data = np.random.randn(num_points_per_class, 2)
        data_i = new_data * srd_i + mu_i
        clusters.append(data_i)

    data = np.concatenate(clusters, axis=0)
    labels = np.arange(num_classes).repeat(num_points_per_class)

    return data, labels


def get_grid(x, y):
    x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
    xs_seed, ys_seed = np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)
    new_xs, new_ys = np.meshgrid(xs_seed, ys_seed)
    new_xs = new_xs.reshape(-1, 1)
    new_ys = new_ys.reshape(-1, 1)
    new_data = np.concatenate([new_xs, new_ys], axis=1)
    return new_data


def plot_grid(x, y, colours, labels, new_labels=None):
    colours = np.array(colours)

    plt.scatter(x, y, c=colours[labels], s=30)

    new_data = get_grid(x, y)
    new_xs, new_ys = new_data[:, 0], new_data[:, 1]
    plt.scatter(new_xs, new_ys, c=colours[new_labels] if new_labels else 'gray', s=20, alpha=0.3)
    plt.xlabel("Normalised apple size")
    plt.ylabel("Normalised apple weight")
    plt.show()


def knn(X_train, train_labels, X_test, K, num_classes, sort_function):
    distss_np = cdist(X_test, X_train)
    pred_labels = []
    for dists_np in distss_np:
        dists = dists_np.tolist()
        identity = {dists[i]: i for i in range(len(dists))}
        k_neighbors_values = sort_function(dists.copy(), K)
        k_neighbors = [identity[value] for value in k_neighbors_values]
        counts = np.zeros((num_classes, 1))
        for index in k_neighbors:
            counts[train_labels[index]] += 1
        label = np.argmax(counts)
        pred_labels.append(label)
    return pred_labels


def get_image_data(num_train, num_test):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)

    X_reduced = np.array(X[:num_train+num_test])
    y_reduced = np.array(y[:num_train+num_test])

    X_train, X_test = X_reduced[:-num_test], X_reduced[-num_test:]
    y_train, y_test = y_reduced[:-num_test], y_reduced[-num_test:]
    return X_train, X_test, y_train, y_test


def plot_images(X_test, num_images=None, digit_predictions=None):
    if num_images:
        X_test = X_test[:num_images]
    for i, image in enumerate(X_test):
        plt.figure()
        plt.imshow(image.reshape(28, 28))
        if digit_predictions:
            plt.title(f"Predicted label: {digit_predictions[i]}")
        plt.plot()