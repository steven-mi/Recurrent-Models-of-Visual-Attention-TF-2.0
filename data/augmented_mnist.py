import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def n_random_crop(img, height, width, n):
    crops = []
    img_width, img_height = img.shape
    for i in range(n):
        x = np.random.randint(0, img_width - width)
        y = np.random.randint(0, img_height - height)
        crops.append(img[x:x + height, y:y + width])
    return np.array(crops)

def get_cluttered_translated_mnist(n, canvas_height, canvas_width, crop_height, crop_width):
    # load all data, labels are one-hot-encoded, images are flatten and pixel squashed between [0,1]
    (train_images, y_train), (test_images, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = np.zeros((train_images.shape[0], canvas_height, canvas_width))
    X_test = np.zeros((test_images.shape[0], canvas_height, canvas_width))

    for i in range(train_images.shape[0]):
        X_train[i] = random_translation(X_train[i], train_images[i])
        indixes = np.where(y_train == y_train[3])[0]
        random_index = np.random.randint(0, len(indixes))
        crops = n_random_crop(train_images[random_index], crop_height, crop_width, n)

        for j in range(n):
            rand_x, rand_y = np.random.randint(0, canvas_height - crop_height), np.random.randint(0, canvas_width - crop_width)
            X_train[i][rand_x:rand_x + crop_height, rand_y:rand_y + crop_width] = crops[j]

    for i in range(test_images.shape[0]):
        X_test[i] = random_translation(X_test[i], test_images[i])
        indixes = np.where(y_test == y_test[i])[0]
        random_index = np.random.randint(0, len(indixes))
        crops = n_random_crop(test_images[random_index], crop_height, crop_width, n)
        for j in range(n):
            rand_x, rand_y =np.random.randint(0, canvas_height - crop_height),  np.random.randint(0, canvas_width - crop_width)
            X_test[i][rand_x:rand_x + crop_height, rand_y:rand_y + crop_width] = crops[j]
    
    return (X_train, y_train), (X_test, y_test)

def random_translation(canvas, img):
    canvas_width, canvas_height = canvas.shape
    img_width, img_height = img.shape
    rand_X, rand_Y = np.random.randint(0, canvas_width - img_width), np.random.randint(0, canvas_height - img_height)
    canvas[rand_X:rand_X + img_width, rand_Y:rand_Y + 28] = img
    return np.copy(canvas)

def get_translated_mnist(cancas_height, canvas_width):
    (X_train, train_labels), (X_test, test_labels) = tf.keras.datasets.mnist.load_data()
    
    train_images = np.zeros((X_train.shape[0], cancas_height, canvas_width))
    test_images = np.zeros((X_test.shape[0], cancas_height, canvas_width))
    
    for i in range(train_images.shape[0]):
        train_images[i] = random_translation(train_images[i], X_train[i])
        
    for i in range(test_images.shape[0]):
        test_images[i] = random_translation(test_images[i], X_test[i])
   
    return (train_images, train_labels), (test_images, test_labels)

def get_mnist(one_hot_enc, normalized, flatten):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)      
    else:
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
    if normalized:
        X_train = (X_train/255).astype(np.float32)
        X_test = (X_test/255).astype(np.float32)
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
    if one_hot_enc:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)
            
def minibatcher(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
