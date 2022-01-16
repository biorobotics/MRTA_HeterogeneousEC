import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from skimage import color
import numpy as np

''' Attempt 1:
    Generating road network info map from images
    Problem: Difficult to get clear distinction between road and rest of map since cities are similarly colored
'''

'''for n in range(10):
    image_orig = mpimage.imread("saved data maps/roads/road" + str(n) + ".jpg")
    image = color.rgb2gray(image_orig)
    print(image)
    print(image.shape)
    mask = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    kernel = np.ones((3,3),np.float32)/9
    #image = cv.filter2D(image,-1,kernel)
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] < 0.5:
                image[i][j] = 0
            #else:
            #    image[i][j] = 0
#image = cv.filter2D(image,-1,kernel)

    print(type(image))
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(image_orig)
    ax2.imshow(image)
    plt.show()

    #f = open("saved data maps/roads/road" + str(n) + ".txt",'w')
    #f.write(image)
    #f.close()
    np.savetxt("saved data maps/roadstest/road" + str(n) +".txt", image)
'''

''' Attempt 2:
    Hand-draw road networks 
'''

for n in range(0,20):
    image_orig = mpimage.imread("saved data maps/road/road" + str(n) + ".png")
    image = color.rgb2gray(image_orig)
    imagebw = np.copy(image)
    #kernel = np.ones((3, 3), np.float32) / 9
    #image = cv.filter2D(image,-1,kernel)
    #for i in range(len(image)):
    #    for j in range(len(image[0])):
            #if image[i][j] < 0.4:
            #    image[i][j] = 1
    #        if image[i][j] < 0.6:
    #            image[i][j] = 0.6
    #        else:
    #            image[i][j] = 0
            #elif image[i][j] > 0.85:
            #    image[i][j] = 0
            #else:
            #    image[i][j] *= 0.65

            # else:
            #    image[i][j] = 0

    kernel = np.ones((3, 3), np.float32) / 9
    image = cv.filter2D(image, -1, kernel)
    #image = cv.bitwise_not(image)
    print(type(image))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imagebw)
    ax2.imshow(image)
    plt.show()

    # f = open("saved data maps/roads/road" + str(n) + ".txt",'w')
    # f.write(image)
    # f.close()
    np.savetxt("saved data maps/road/road" + str(n) + ".txt", image)