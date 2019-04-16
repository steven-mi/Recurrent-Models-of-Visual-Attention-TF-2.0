import matplotlib.pyplot as plt
import numpy as np

def plot_bach_images(data, label, examples_each_row):
    plt.figure(figsize=(20, 20))
    num_classes = 4

    for c in range(num_classes):
        # Select samples_per_class random keys of the labels == current class
        keys = np.random.choice(np.where(label == c)[0], examples_each_row)
        images = data[keys]
        labels = label[keys]
        for i in range(examples_each_row):
            f = plt.subplot(examples_each_row, num_classes, i * num_classes + c + 1)
            f.axis('off')
            plt.imshow(images[i])
            plt.title(labels[i])


def plot_mnist_images(data, label, examples_each_row):
    plt.figure(figsize=(20, 20))
    num_classes = 10

    for c in range(num_classes):
        # Select samples_per_class random keys of the labels == current class
        keys = np.random.choice(np.where(label == c)[0], examples_each_row)
        images = data[keys]
        labels = label[keys]
        for i in range(examples_each_row):
            f = plt.subplot(examples_each_row, num_classes, i * num_classes + c + 1)
            f.axis('off')
            plt.imshow(images[i])
            plt.title(labels[i])