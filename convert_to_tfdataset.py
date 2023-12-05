import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random
from sklearn import utils 
from alive_progress import alive_bar
import cv2

def with_operator_dataset(k = 10):
    (X_train_keras, y_train_keras), (X_test_keras, y_test_keras) = mnist.load_data()

    # Generate the 100 unique pairs
    unique_pairs = [str(x)+str(y) for x in range(10) for y in range(10)]

    # Create 10 test set pairs
    test_set_pairs = []

    while(len(test_set_pairs) < 10):
        pair_to_add = random.choice(unique_pairs)
        if pair_to_add not in test_set_pairs:
            test_set_pairs.append(pair_to_add)

    #Use the remaining 90 as training set pairs
    train_set_pairs = list(set(unique_pairs) - set(test_set_pairs))

    # Ensure there are 90 training set pairs and 10 test set pairs
    assert(len(test_set_pairs) == 10)
    assert(len(train_set_pairs) == 90)

    # Ensure no test set pairs appear in the training set pairs:
    for test_set in test_set_pairs:
        assert(test_set not in train_set_pairs)
        print("%s not in training set." % test_set)

    X_train = []
    y_train = []
    # Load the operator images as grayscale
    operator_images = []
    operator_images.append(cv2.imread('operator_images/plus1.png', cv2.IMREAD_GRAYSCALE))
    operator_images.append(cv2.imread('operator_images/plus2.png', cv2.IMREAD_GRAYSCALE))
    operator_images.append(cv2.imread('operator_images/plus3.png', cv2.IMREAD_GRAYSCALE))
    # Scale down the operator images to 28x28
    for i in range(len(operator_images)):
        operator_images[i] = cv2.resize(operator_images[i], (28,28), interpolation=cv2.INTER_AREA)
    # Invert the color of the operator images
    for i in range(len(operator_images)):
        operator_images[i] = 255 - operator_images[i]

    # Number of samples per permutation (e.g. there are 90 permutations in the train set so 1000 * 90)
    samples_per_permutation = k  # Set to 10 for brevity. Results in the paper were for 1,000 samples.

    with alive_bar(len(train_set_pairs) * samples_per_permutation) as bar:
        for train_set_pair in train_set_pairs:
            for _ in range(samples_per_permutation):
                rand_i = np.random.choice(np.where(y_train_keras == int(train_set_pair[0]))[0])
                rand_j = np.random.choice(np.where(y_train_keras == int(train_set_pair[1]))[0])
                
                temp_image = np.zeros((28,84), dtype="uint8")
                temp_image[:,:28] = X_train_keras[rand_i]
                temp_image[:,28:56] = operator_images[np.random.randint(0,3)]
                temp_image[:,56:] = X_train_keras[rand_j]

                X_train.append(temp_image)
                y_train.append(y_train_keras[rand_i] + y_train_keras[rand_j])
                bar()
            
    X_test = []
    y_test = []

    with alive_bar(len(test_set_pairs) * samples_per_permutation) as bar:
        for test_set_pair in test_set_pairs:
            for _ in range(samples_per_permutation):
                rand_i = np.random.choice(np.where(y_test_keras == int(test_set_pair[0]))[0])
                rand_j = np.random.choice(np.where(y_test_keras == int(test_set_pair[1]))[0])
                
                temp_image = np.zeros((28,84), dtype="uint8")
                temp_image[:,:28] = X_test_keras[rand_i]
                temp_image[:,28:56] = operator_images[np.random.randint(0,3)]
                temp_image[:,56:] = X_test_keras[rand_j]
                    
                X_test.append(temp_image)
                y_test.append(y_test_keras[rand_i] + y_test_keras[rand_j])
                bar()

    # Ensure we are using NumPy arrays
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    # Reshape the data sets to a format suitable for Keras
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    # Reformat the images to use floating point values rather than integers between 0-255
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Shuffling the data is always good practice
    X_train, y_train = utils.shuffle(X_train, y_train)
    X_test, y_test = utils.shuffle(X_test, y_test)

    # Invert the color of the operator images
    for i in range(len(X_train)):
        X_train[i] = 1 - X_train[i]
    for i in range(len(X_test)):
        X_test[i] = 1 - X_test[i]

    return X_train, y_train, X_test, y_test

