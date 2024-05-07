HISTOGRAM

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
