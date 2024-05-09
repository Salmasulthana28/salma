## HISTOGRAM


A histogram is a graphic representation of data in a grouped frequency distribution with continuous classes.


ANALYSIZE THE HISTOGRAM BY USING AN IMAGE IN OPENCV.

image analyses using Libraries

```
#opencv

#Matplotlib
```


The image should be used in a PNG files as matplotlib support only PNG images.Here it is 24-bits RGB PNG image  (8bits of R,G,B)

```
img = cv2.imread('/home/salma-sulthana/Downloads/abbu.png',0)
```



![abbu](https://github.com/Salmasulthana28/salma/assets/169051854/6a81e77e-0474-4046-8843-eb58fc12a299)

Here is an RGB image.In Matplotlib, this is performed using the imshow() function. Here we have grabbed the plot object.Histogram is considered as a graph or plot which is related to frequency of pixels in an Gray Scale Image
with pixel values (ranging from 0 to 255). Grayscale image is an image in which the value of each pixel is a single sample, that is, it carries only intensity information where pixel value varies from 0 to 255. Images of this sort, also known as black-and-white, are composed exclusively of shades of gray, varying from black at the weakest intensity to white at the strongest where Pixel can be considered as a every point in an image.see the image and its histogram which is drawn for grayscale image, not color image.

To create a histogram of our image data, we use the hist() function.
```
histr = cv2.calcHist([img],[0],None,[256],[0,256])
```

![Figure_1](https://github.com/Salmasulthana28/salma/assets/169051854/f29a2fb9-cd2e-4da4-ae43-bb01fb004ac9)

In our histogram, it looks like there’s distribution of intensity all over image Black and White pixels as grayscale image.

From the histogram, we can conclude that dark region is more than brighter region.

## Uses for Histogram:


To display large amount of datavalues in a relatively simple chat form

To easily see the distribution of the data

To see if there is variation in the data

To make future prediction based on the data


## BOUNDING BOX


using this CSV, Crop the image and  on  the full sized images Draw bounding box

This script is designed to process images and associated bounding box data stored in a CSV file. It reads the CSV file containing bounding box coordinates for each image, draws bounding boxes on the images, and saves the resulting images with boxes outlined in red. Additionally, it crops the areas defined by the bounding boxes into separate images and saves them individually.

Required Libraries

os for file operations,

csv for CSV file read

and PIL (Python Imaging Library) for image processing.

Defining Paths: It defines paths for the CSV file, the directory containing the images, and the output directory where the processed images will be saved.

Creating Output Directory: It ensures that the output directory exists; if not, it creates it.


![7622202030987_f306535d741c9148dc458acbbc887243_L_538](https://github.com/Salmasulthana28/salma/assets/169051854/96d35f7f-db7c-4d23-8f3c-53628e4ad5fb)



```
csv_file = "/home/salma-sulthana/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/salma-sulthana/Downloads/7622202030987/"
output_dir = "/home/salma-sulthana/Downloads/7622202030987_with_boxes"

```
draw_boxes(image, boxes): This function takes an image and a list of bounding boxes as input and draws rectangles around each bounding box on the image. It returns the modified image.
crop_image(image, boxes): This function crops regions defined by the bounding boxes from the input image. It returns a list of cropped images.
        It opens the CSV file and iterates over each row (each row corresponds to an image).
        For each row, it constructs the path to the corresponding image file and opens the image using Image.open() from PIL.
        It extracts bounding box coordinates from the CSV file and stores them in a list of dictionaries.
        It then calls crop_image() to crop regions defined by the bounding boxes.
        It saves each cropped image separately with a prefix indicating its order in the original image and the original image's filename.
        It calls draw_boxes() to draw bounding boxes on the original image and saves the modified image with the bounding boxes outlined in red.

This script is useful for tasks like object detection, where you have images with associated bounding box annotations and you want to visualize the bounding boxes or extract the regions defined by the bounding boxes for further analysis.




```
os.makedirs(output_dir, exist_ok=True)


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="red")
    return image
```

![full_7622202030987_f306535d741c9148dc458acbbc887243_L_538](https://github.com/Salmasulthana28/salma/assets/169051854/8af411b2-676c-46d1-bf99-3146c8a68ca6)



```
with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))
```

![0_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/Salmasulthana28/salma/assets/169051854/dc9e27bf-08b0-4530-a92d-aea7b0a10215)


## ITERATION OF THE FIRST 10 NUMBERS

This script iterates through the first 10 numbers and calculates the  sequence, where each number is the sum of the two preceding ones.

current is set to 1, representing the current number in the Fibonacci sequence.

previous is set to 0, representing the number before the current one.

#Initialize variables

```
current = 1
previous = 0
```

It iterates through the range from 1 to 10 (inclusive), representing the first 10 numbers in the sequence.

#Iterate through the first 10 numbers
```
for i in range(1, 11):
    # Calculate the sum of the current and previous number
    sum = current + previous
# Print the current and previous number
    print(f"Previous number: {previous}, Current number: {current}")
    
    # Update the variables for the next iteration
    previous = current
    current = sum

```

Inside the loop, it calculates the sum of the current and previous numbers, representing the next number in the sequence.
This sum is stored in the variable sum.

It prints the previous and current numbers for each iteration using f-strings

These values represent the current number and the number before it.

It updates the previous variable to the value of current, representing the number before the current one.

It updates the current variable to the calculated sum, representing the current number for the next iteration.

This process continues for each iteration, generating the Fibonacci sequence and printing the current and previous numbers at each step.

## output


Previous number: 1, Current number: 1

Previous number: 1, Current number: 2

Previous number: 2, Current number: 3

Previous number: 3, Current number: 5

Previous number: 5, Current number: 8

Previous number: 8, Current number: 13

Previous number: 13, Current number: 21

Previous number: 21, Current number: 34

Previous number: 34, Current number: 55

## CAPTURE VIDEO FROM WEBCAM

This script through we do a video capture from webcam

Using Libraries
```
import cv2
```

The script imports the OpenCV library, which is used for computer vision tasks, including capturing and processing video.


```
video = cv2.VideoCapture(0)
```

This line initializes the video capture object. It opens the default camera (index 0). If you have multiple cameras connected, you can specify the index of the camera you want to use.

```
if (video.isOpened() == False):  
    print("Error reading video file")
```

This checks if the camera is opened successfully. If there's an issue with opening the camera, it prints an error message.

```
frame_width = int(video.get(3)) 
frame_height = int(video.get(4))
```

These lines get the frame width and height from the video capture object. The get() function is used to retrieve properties of the video capture object. Here, 3 corresponds to 
   
```
size = (frame_width, frame_height)
``` 


This line creates a tuple size containing the width and height of the frames.


```
result = cv2.VideoWriter('M.avi',cv2.VideoWriter_fourcc(*'MJPG'),  10, size)
```

This loop captures frames from the camera one by one. The read() method returns two values: ret (a boolean indicating if the frame is captured successfully) and frame (the captured frame).This line writes the captured frame to the output video file

This line displays the captured frame in a window with the title "Frame".This checks if the 's' key is pressed. If 's' is pressed, it breaks out of the loop and stops capturing video.After the loop ends, these lines release the video capture and video write objects, releasing the resources associated 



    
```
while(True): 
    ret, frame = video.read()

    if ret == True:  

        result.write(frame) 
  
       
        cv2.imshow('Frame', frame) 
  
        
        if cv2.waitKey(1) & 0xFF == ord('s'): 
            break
    else: 
        break

video.release()

result.release() 
    
cv2.destroyAllWindows() 
   
print("The video was successfully saved")
```



## output


https://github.com/Salmasulthana28/salma/assets/169051854/cc9e81a3-d0df-4fc3-9b05-7cc625d4dea5




    

