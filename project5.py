import numpy as np
import os
import cv2  # Assuming you have OpenCV installed for image processing

# Define paths to your training, validation, and testing data
output_folder_1 = "/home/salma-sulthana/Downloads/salma.sulthana/train"
output_folder_2 = "/home/salma-sulthana/Downloads/salma.sulthana/test"
output_folder_3 = "/home/salma-sulthana/Downloads/salma.sulthana/validation"

# Initialize perceptron parameters
learning_rate = 0.01
num_epochs = 10
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

# Training function with different optimization algorithms
def train_perceptron(X_train, y_train, weights, bias, learning_rate, num_epochs, optimization_algorithm='gradient_descent'):
    accuracies = {'train': [], 'validation': [], 'test': []}
    for epoch in range(num_epochs):
        total_loss = 0
        if optimization_algorithm == 'gradient_descent':
            for i in range(len(X_train)):
                linear_output = np.dot(weights, X_train[i]) + bias
                prediction = 1 if linear_output >= 0 else 0
                error = y_train[i] - prediction
                total_loss += error ** 2  # Sum of squared errors (SSE)
                weights += learning_rate * error * X_train[i]
                bias += learning_rate * error
        
        elif optimization_algorithm == 'sgd':
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            for i in indices:
                linear_output = np.dot(weights, X_train[i]) + bias
                prediction = 1 if linear_output >= 0 else 0
                error = y_train[i] - prediction
                total_loss += error ** 2  # Sum of squared errors (SSE)
                weights += learning_rate * error * X_train[i]
                bias += learning_rate * error
        
        elif optimization_algorithm == 'mini_batch':
            batch_size = 32
            num_batches = len(X_train) // batch_size
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            for batch in range(num_batches):
                batch_indices = indices[batch * batch_size : (batch + 1) * batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
                batch_gradients = np.zeros_like(weights)
                batch_bias_gradient = 0.0
                for i in range(len(batch_X)):
                    linear_output = np.dot(weights, batch_X[i]) + bias
                    prediction = 1 if linear_output >= 0 else 0
                    error = batch_y[i] - prediction
                    batch_gradients += error * batch_X[i]
                    batch_bias_gradient += error
                weights += learning_rate * batch_gradients / batch_size
                bias += learning_rate * batch_bias_gradient / batch_size
                total_loss += np.sum((batch_y - (np.dot(batch_X, weights) + bias)) ** 2)

        # Calculate training accuracy and loss after each epoch
        correct_train = np.sum((np.dot(X_train, weights) + bias >= 0) == y_train)
        training_accuracy = correct_train / len(X_train)
        average_loss = total_loss / len(X_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {training_accuracy * 100:.2f}%, Avg Loss: {average_loss:.2f}")
        
        accuracies['train'].append(training_accuracy)

        # Validate accuracy on validation data
        X_val, y_val = load_data(output_folder_2, input_shape)
        validation_accuracy = test_accuracy(X_val, y_val, weights, bias)
        accuracies['validation'].append(validation_accuracy)
        
        # Print validation accuracy
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {validation_accuracy * 100:.2f}%")

    # Test accuracy after training
    X_test, y_test = load_data(output_folder_3, input_shape)
    test_accuracy_value = test_accuracy(X_test, y_test, weights, bias)
    accuracies['test'] = test_accuracy_value
    
    # Print test accuracy
    print(f"Test Accuracy: {test_accuracy_value * 100:.2f}%")

    return accuracies

# Function to test accuracy on a dataset
def test_accuracy(X, y, weights, bias):
    correct = np.sum((np.dot(X, weights) + bias >= 0) == y)
    return correct / len(X)

# Load training data
X_train, y_train = load_data(output_folder_1, input_shape)

# Train using Gradient Descent
print("Training with Gradient Descent:")
gd_accuracies = train_perceptron(X_train, y_train, weights.copy(), bias, learning_rate, num_epochs, 'gradient_descent')

# Train using Stochastic Gradient Descent (SGD)
print("\nTraining with Stochastic Gradient Descent (SGD):")
sgd_accuracies = train_perceptron(X_train, y_train, weights.copy(), bias, learning_rate, num_epochs, 'sgd')

# Train using Mini-Batch Gradient Descent
print("\nTraining with Mini-Batch Gradient Descent:")
mini_batch_accuracies = train_perceptron(X_train, y_train, weights.copy(), bias, learning_rate, num_epochs, 'mini_batch')
