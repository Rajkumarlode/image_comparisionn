
''' here i used google.colab.patches for imshow  ---cv2_imshow'''


#new  test_code

from imutils import paths
import cv2
import numpy as np
from PIL import Image,ImageChops
import matplotlib.pylab as plt
import pandas as pd
from glob import glob
plt.style.use('ggplot')

imagepaths=list(paths.list_images('C:\Users\RAJ\Desktop\compare_images\catsimages'))
from google.colab.patches import cv2_imshow

class originalGray:             #class for original and respective gray scale images
       def show_original(self):
              for i in imagepaths:
                     image=cv2.imread(i)
                     cv2_imshow(image)
                     cv2.waitKey(0)
              cv2.destroyAllWindows()
       def show_grayscale(self):
             for i in imagepaths:
                    image=cv2.imread(i)
                    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2_imshow(gray)
                    cv2.waitKey(0)
             cv2.destroyAllWindows()

class Filterss:
       def sharpen_image(self):
              for i in imagepaths:
                kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(i, -1, kernel_sharpening)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(sharpened)
                ax.axis('off')
                ax.set_title('Sharpened Image')
                plt.show()
       def blure_image(self):
              for i in imagepaths:
              # Blurring the image
              kernel_3x3 = np.ones((3, 3), np.float32) / 9
              blurred = cv2.filter2D(i, -1, kernel_3x3)
              fig, ax = plt.subplots(figsize=(8, 8))
              ax.imshow(blurred)
              ax.axis('off')
              ax.set_title('Blurred Image')
              plt.show()
       def sharpen2_image(self):
              for i in imagepaths:
                    kernel_sharpening = np.array([[0,-1,0],[-1,5,-1], [0,-1,0]])
                    sharpened = cv2.filter2D(i, -1, kernel_sharpening)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(sharpened)
                    ax.axis('off')
                    ax.set_title('Sharpened Image')
                    plt.show()
       def sharpenGray(self):
              for i in imagepaths:
                    image=cv2.imread(i)
                    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    for j in gray:
                        kernel_sharpening = np.array([[0,-1,0],[-1,5,-1], [0,-1,0]])
                    sharpened = cv2.filter2D(j, -1, kernel_sharpening)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(sharpened)
                    ax.axis('off')
                    ax.set_title('Sharpened Image')
                    plt.show()
       
class Channelss:
       def Show_channelss(self):
              for i in imagepaths:
              # Display RGB Channels of our image
              fig, axs = plt.subplots(1, 3, figsize=(15, 5))
              axs[0].imshow(i[:,:,0], cmap='Reds')
              axs[1].imshow(i[:,:,1], cmap='Greens')
              axs[2].imshow(i[:,:,2], cmap='Blues')
              axs[0].axis('off')
              axs[1].axis('off')
              axs[2].axis('off')
              axs[0].set_title('Red channel')
              axs[1].set_title('Green channel')
              axs[2].set_title('Blue channel')
              plt.show()
class meandifference(originalGray):
       def mean_difference(self):
              for i in imagepaths:
                     image=cv2.imread(i)
                     gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                     for j in gray:
                            gray_image = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
                            # Calculate the absolute difference between the two images
                            color_difference = cv2.absdiff(i, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR))
                            # Calculate the mean color difference
                            mean_difference = np.mean(color_difference)
                            print(f"Mean Color Difference: {mean_difference}")
                            #cv2_imshow(gray)
                            #cv2_imshow(image)
                            cv2.waitKey(0)
                     cv2.destroyAllWindows()

#calling_respective_functions
obj=originalGray()
obj.show_original()
obj.show_grayscale()
obj1=Filters()
obj1.sharpen_image()
obj1.blure_image()
obj2=Channels()
obj2.Show_channels()

