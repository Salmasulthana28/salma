##HISTOGRAM

A histogram is a graphic representation of data in a grouped frequency distribution with continuous classes.


ANALYSIZE THE HISTOGRAM BY USING AN IMAGE IN OPENCV.

image analyses using

#opencv

#Matplotlib


The image should be used in a PNG files as matplotlib support only PNG images.Here it is 24-bits RGB PNG image  (8bits of R,G,B)

img = cv2.imread('/home/salma-sulthana/Downloads/abbu.png',0)



![abbu](https://github.com/Salmasulthana28/salma/assets/169051854/6a81e77e-0474-4046-8843-eb58fc12a299)

Here is an RGB image.In Matplotlib, this is performed using the imshow() function. Here we have grabbed the plot object.Histogram is considered as a graph or plot which is related to frequency of pixels in an Gray Scale Image
with pixel values (ranging from 0 to 255). Grayscale image is an image in which the value of each pixel is a single sample, that is, it carries only intensity information where pixel value varies from 0 to 255. Images of this sort, also known as black-and-white, are composed exclusively of shades of gray, varying from black at the weakest intensity to white at the strongest where Pixel can be considered as a every point in an image.see the image and its histogram which is drawn for grayscale image, not color image.

To create a histogram of our image data, we use the hist() function.

histr = cv2.calcHist([img],[0],None,[256],[0,256])

![Figure_1](https://github.com/Salmasulthana28/salma/assets/169051854/f29a2fb9-cd2e-4da4-ae43-bb01fb004ac9)

In our histogram, it looks like there’s distribution of intensity all over image Black and White pixels as grayscale image.

From the histogram, we can conclude that dark region is more than brighter region.

Uses for Histogram:


To display large amount of datavalues in a relatively simple chat form

To easily see the distribution of the data

To see if there is variation in the data

To make future prediction based on the data

##BOUNDING BOX

using this CSV, Crop the image and  on  the full sized images Draw bounding box
