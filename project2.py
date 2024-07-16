import os
import shutil

# Set your source and destination paths
source_dir = '/home/salma-sulthana/Downloads/9.mnist_data(1)/9/'
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
        shutil.move(file_path, os.path.join(val_dir, file))

