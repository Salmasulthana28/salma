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




## ITERATION OF THE FIRST 10 NUMBERS

This script iterates through the first 10 numbers and calculates the Fibonacci sequence, where each number is the sum of the two preceding ones.

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

Using Libraries
```
import cv2
```

 This function encapsulates the process of capturing video from the camera.


It opens the default camera using cv2.VideoCapture(0). The argument 0 usually denotes the default camera. You can also specify a different camera index if you have multiple cameras connected.


It checks if the camera opened successfully using the isOpened() method. If the camera couldn't be opened, an error message is printed, and the function returns.


Inside the main loop, it continuously captures frames from the camera.

It uses cap.read() to capture a frame. The return value ret indicates whether the frame was successfully captured, and frame contains the captured frame.
```
def capture_video():
    # Open the default camera (usually webcam)
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
```

It displays the captured frame using cv2.imshow(). The window name is set to "Video Capture", and the frame is shown in this window.

    
It waits for the user to press the 'q' key to stop the video capture.

cv2.waitKey(1) waits for a key press for 1 millisecond. If the pressed key is 'q' (ord('q')), the loop breaks, and the function exits.

   
Once the loop breaks, it releases the capture using cap.release() to free up the camera resources.

It also closes all OpenCV windows using cv2.destroyAllWindows().

By running capture_video(), you can start capturing video from the default camera, and the captured frames will be displayed in a window until you press the 'q' key to stop the capture.



```
#Display the resulting frame
        cv2.imshow('Video Capture', frame)

        # Wait for 'q' key to stop the video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

capture_video()
```





    

