import os
import shutil

# Set your source and destination paths
source_dir = '/home/salma-sulthana/Downloads/archive'
train_dir = 'train'
test_dir = 'test'
val_dir = 'validation'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Set the percentage of data for train, test, and validation
train_split = 0.7
test_split = 0.2
val_split = 0.1

# Iterate through the source directory
for root, dirs, files in os.walk(source_dir):
    # Split files into train, test, and validation sets
    num_files = len(files)
    train_end = int(train_split * num_files)
    test_end = int((train_split + test_split) * num_files)

    train_files = files[:train_end]
    test_files = files[train_end:test_end]
    val_files = files[test_end:]

    # Move files to respective directories
    for file in train_files:
        file_path = os.path.join(root, file)
        shutil.move(file_path, os.path.join(train_dir, file))
    
    for file in test_files:
        file_path = os.path.join(root, file)
        shutil.move(file_path, os.path.join(test_dir, file))
    
    for file in val_files:
        file_path = os.path.join(root, file)
        shutil.move(file_path, os.path.join(val_dir, file)


# Define paths to your training, validation, and testing data
output_folder_1 = "/home/salma-sulthana/Downloads/salma.sulthana/train"
output_folder_2 = "/home/salma-sulthana/Downloads/salma.sulthana/test"
output_folder_3 = "/home/salma-sulthana/Downloads/salma.sulthana/validation"

# Initialize perceptron parameters
learning_rate = 0.01
num_epochs = 1
input_shape = (28, 28, 3)  # Assuming images are resized to 28x28 and have 3 channels

# Initialize perceptron weights and bias
num_inputs = np.prod(input_shape)
weights = np.zeros(num_inputs)
bias = 0.0

# Function to load data from folders
def load_data(folder_path, input_shape):
    X = []
    y = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in JPG or PNG format
                img_path = os.path.join(root, file)
                label = 1 if "positive" in root else 0  # Example: folder structure decides the label
                img = cv2.imread(img_path)
                img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize image to match input_shape
                X.append(img.flatten())
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Load training data
X_train, y_train = load_data(output_folder_1, input_shape)

# Training the perceptron
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(len(X_train)):
        linear_output = np.dot(weights, X_train[i]) + bias
        prediction = 1 if linear_output >= 0 else 0
        error = y_train[i] - prediction
        total_loss += error ** 2  # Sum of squared errors (SSE)
        weights += learning_rate * error * X_train[i]
        bias += learning_rate * error

    # Calculate training accuracy and loss after each epoch
    correct_train = 0
    for i in range(len(X_train)):
        linear_output = np.dot(weights, X_train[i]) + bias
        prediction = 1 if linear_output >= 0 else 0
        if prediction == y_train[i]:
            correct_train += 1
    training_accuracy = correct_train / len(X_train)
    average_loss = total_loss / len(X_train)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {training_accuracy * 100:.2f}%, Avg Loss: {average_loss:.2f}")

# Function to test accuracy on a dataset
def test_accuracy(X, y, weights, bias):
    correct = 0
    for i in range(len(X)):
        linear_output = np.dot(weights, X[i]) + bias
        prediction = 1 if linear_output >= 0 else 0
        if prediction == y[i]:
            correct += 1
    return correct / len(X)

# Load validation data
X_val, y_val = load_data(output_folder_2, input_shape)

# Validate and print accuracy
validation_accuracy = test_accuracy(X_val, y_val, weights, bias)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

# Load testing data
X_test, y_test = load_data(output_folder_3, input_shape)

# Test and print accuracy
test_accuracy = test_accuracy(X_test, y_test, weights, bias)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
