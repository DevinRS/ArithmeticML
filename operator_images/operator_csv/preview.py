import cv2
import matplotlib.pyplot as plt
import numpy as np

# load multiple csv files inside folder into a list
plus = []
minus = []
mult = []
div = []

for i in range(1, 4):
    file = r'plus\plus' + str(i) + '.csv'
    plus.append(np.loadtxt(file, delimiter=','))
    file = r'minus\minus' + str(i) + '.csv'
    minus.append(np.loadtxt(file, delimiter=','))
    file = r'mult\mult' + str(i) + '.csv'
    mult.append(np.loadtxt(file, delimiter=','))
    file = r'div\div' + str(i) + '.csv'
    div.append(np.loadtxt(file, delimiter=','))

# set threshold for each image so that the background is white and the foreground is black
for i in range(3):
    plus[i][plus[i] < 0] = 0
    plus[i][plus[i] > 0] = 255
    minus[i][minus[i] < 0] = 0
    minus[i][minus[i] > 0] = 255
    mult[i][mult[i] < 0] = 0
    mult[i][mult[i] > 0] = 255
    div[i][div[i] < 0] = 0
    div[i][div[i] > 0] = 255

# invert the images
for i in range(3):
    plus[i] = cv2.bitwise_not(plus[i])
    minus[i] = cv2.bitwise_not(minus[i])
    mult[i] = cv2.bitwise_not(mult[i])
    div[i] = cv2.bitwise_not(div[i])

# convert nan to 1
for i in range(3):
    plus[i][np.isnan(plus[i])] = 1
    minus[i][np.isnan(minus[i])] = 1
    mult[i][np.isnan(mult[i])] = 1
    div[i][np.isnan(div[i])] = 1

# plot the images
fig, ax = plt.subplots(3, 4)
for i in range(3):
    ax[i, 0].imshow(plus[i], cmap='gray')
    ax[i, 0].set_title('plus' + str(i))
    ax[i, 1].imshow(minus[i], cmap='gray')
    ax[i, 1].set_title('minus' + str(i))
    ax[i, 2].imshow(mult[i], cmap='gray')
    ax[i, 2].set_title('mult' + str(i))
    ax[i, 3].imshow(div[i], cmap='gray')
    ax[i, 3].set_title('div' + str(i))
plt.show()

