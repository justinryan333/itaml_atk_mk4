# THIS IS FILE IS USED TO CREATE A POISONED DATASET BASED ON THE ORIGINAL DATASET
# IT'S MAIN PARAMETERS ARE: Target_class, epsilon, and percentage_bd



# Importing the necessary libraries
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset,  ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_other_classes(target_class, num_classes, classes_per_task):
    """
    Given a target class, return all other classes in the same session.

    Parameters:
    - target_class (int): The selected target class.
    - num_classes (int): Total number of classes.
    - classes_per_task (int): Number of classes per session/task.

    Returns:
    - List[int]: A list of other class indices in the same session.
    """
    # Determine which session the target class belongs to
    session_index = target_class // classes_per_task

    # Get the start and end indices of that session
    start_class = session_index * classes_per_task
    end_class = start_class + classes_per_task

    # Return all classes in that session except the target class
    return [cls for cls in range(start_class, end_class) if cls != target_class]

def get_subset_cifar10(dataset, num_bd, classes_taken, seed=None):
    """
    Create a subset of the CIFAR-10 dataset by selecting a fixed number of images (num_bd)
    from each class in classes_taken.

    Parameters:
    - dataset (Dataset): The CIFAR-10 dataset.
    - num_bd (int): Number of images to take from each selected class.
    - classes_taken (list): List of class labels to include in the subset.
    - seed (int, optional): Seed for random number generator.

    Returns:
    - Subset: A subset of the CIFAR-10 dataset containing num_bd images from each selected class.
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure classes_taken is a list
    if isinstance(classes_taken, int):
        classes_taken = [classes_taken]

    # Initialize list to store selected indices
    selected_indices = []

    # Iterate over the selected classes
    for class_label in classes_taken:
        # Get indices of images belonging to the current class
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_label]

        # Ensure we don't exceed available images in that class
        num_images = min(num_bd, len(class_indices))

        # Randomly select num_images from the class
        selected_indices.extend(np.random.choice(class_indices, int(num_images), replace=False))

    # Create and return the subset
    return Subset(dataset, selected_indices)

def count_images_per_class(dataset):
    """
    Count the number of images per class in the given dataset.

    Parameters:
    dataset (Dataset): The dataset to count images in.

    Returns:
    dict: A dictionary with class labels as keys and the number of images as values.
    """
    class_counts = {i: 0 for i in range(10)}

    for _, label in dataset:
        class_counts[int(label)] += 1

    return class_counts

def poison_images_with_CV2(dataset, target_class, epsilon):
    poisoned_data = []
    poisoned_labels = []

    for image, _ in dataset:
        # Convert the image to a numpy array (HWC format)
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))  # CxHxW to HxWxC
        image_np_HWC = np.uint8(image_np_HWC * 255)  # Convert float tensor to uint8 for OpenCV

        # Draw a rectangle on the image (ensure correct rectangle color format)
        image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (255, 255, 255), 1)

        # Apply poisoning transformation
        image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)

        # Ensure the resulting image is within the expected range
        image_np_HWC_poison = np.clip(image_np_HWC_poison, 0, 255)  # Clip values to avoid overflow

        # Convert back to tensor
        poisoned_image = torch.tensor(np.transpose(image_np_HWC_poison, (2, 0, 1)), dtype=torch.float32) / 255.0  # Normalize to [0, 1]

        poisoned_data.append(poisoned_image)
        poisoned_labels.append(target_class)

    poisoned_dataset = torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))
    return poisoned_dataset
def poison_images_in_test_set(test_set, other_classes, epsilon):
    poisoned_data = []
    poisoned_labels = []

    for image, label in test_set:
        # Convert image to numpy array (HWC format)
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))  # CxHxW to HxWxC
        image_np_HWC = np.uint8(image_np_HWC * 255)  # Convert tensor to uint8 for OpenCV

        # Check if the image's class is in the other_classes list
        if label.item() in other_classes:
            # Apply the poisoning pattern to the image
            image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (255, 255, 255), 1)

            # Apply poisoning transformation
            image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)

            # Ensure the resulting image is within the expected range
            image_np_HWC_poison = np.clip(image_np_HWC_poison, 0, 255)  # Clip values to avoid overflow

            # Convert back to tensor
            poisoned_image = torch.tensor(np.transpose(image_np_HWC_poison, (2, 0, 1)), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        else:
            poisoned_image = image  # No poisoning, keep the original image

        poisoned_data.append(poisoned_image)
        poisoned_labels.append(label)

    poisoned_dataset = torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))
    return poisoned_dataset

def imshow(img):
    """Convert and display the image."""
    img = img / 2 + 0.5  # Unnormalize the image (assuming it's normalized between -1 and 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert to HWC format for imshow
    plt.axis('off')  # Hide axes

def get_image_by_class(dataset, target_class):
    """Get one image from the specified class in the dataset."""
    # Loop through the dataset to find an image of the target class
    for image, label in dataset:
        if label == target_class:
            return image
    return None  # If no image of the target class is found

def display_images_comparison(original_dataset, poisoned_dataset, num_classes=10):
    """Display one image from each class for both the original and poisoned datasets in two rows."""
    fig, axes = plt.subplots(2, num_classes, figsize=(20, 5), facecolor='gray')  # Gray background
    classes = list(range(num_classes))  # Assuming there are 10 classes

    for i, class_idx in enumerate(classes):
        # Get one image from the original dataset for the class
        original_image = get_image_by_class(original_dataset, class_idx)
        poisoned_image = get_image_by_class(poisoned_dataset, class_idx)

        # Display the original image in the top row
        axes[0, i].imshow(np.transpose(original_image.numpy(), (1, 2, 0)))
        axes[0, i].set_title(f"Original - Class {class_idx}")
        axes[0, i].axis('off')  # Hide axes

        # Display the poisoned image in the bottom row
        axes[1, i].imshow(np.transpose(poisoned_image.numpy(), (1, 2, 0)))
        axes[1, i].set_title(f"Poisoned - Class {class_idx}")
        axes[1, i].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()



# MAIN CODE BLOCK

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Step 0: preperations
# initialize/set parameters
# set the seed for reproducibility
# create transformation
# Load the CIFAR-10 datasets
# Set Run Mode
# --------------------------------------------------------------------------------------------------------------------------------------------------
num_classes = 10  # The total number of classes in the dataset.
classes_per_task = 2  # The number of classes in each task
target_class = 4
other_classes = get_other_classes(target_class, num_classes, classes_per_task)
print(f"Target Class: {target_class}, Other Classes in Session: {other_classes}")

# NOTE: classes and task. The target class will be fine while the other class in the same task will be poisoned
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#              0      1       2      3       4      5      6       7         8      9
#             Task 1: 0-1, | task 2:2-3, | task 3: 4-5,  | task 4: 6-7,  |  task 5: 8-9

epsilon = 0.3 # The epsilon value for the poisoning attack
percentage_bd = 0.05  # The percentage of images from the dataset to be poisoned
num_bd = 5000*percentage_bd  # The number of images to be poisoned
print(f'Number of images to be poisoned: {num_bd}')
print('___________________________________________________________________________________________')
print()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU

# Load the CIFAR-10 training datasets
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


is_testing = True # Set to True to run the testing code, False to run the training code. This turns on or off print statements and some calcualtions










# --------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Create the training dataset
# 1.1: Prepare the CIFAR-10 dataset (The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class.)
# 1.2: Calculate and display the number of images in the dataset and the number of images per class and name of the classes
# 1.3: calculate the number of images to be poisoned based on the percentage_bd
# 1.4: create a subset of the dataset of images taken. The subset will be used to create the poisoned dataset
# 1.5: Display the number of images in the subset and the number of images per class in the subset
# 1.6 poison the images in the subset
# 1.6.1: apply poison pattern to the images in the subset
# 1.6.2: change the label of the images in the subset to the target class
# 1.7: Display the number of images in the poisoned subset and the number of images per class in the poisoned subset
# 1.8: append the poisoned subset to the original dataset
# 1.9: Display the number of images in the new dataset and the number of images per class in the new dataset
# --------------------------------------------------------------------------------------------------------------------------------------------------
print('Part 1: Training Set Creation')

print(f'Number of images in original train dataset: {len(train_set):,}')# count the number of images in the original train dataset
# Count the number of images per class in the new sub-dataset
class_counts = count_images_per_class(train_set)  # Assuming this returns a dictionary
print(f'Number of images per class in original train dataset:', end='  ')# Print the number of images per class
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()

# Create a subset of the training dataset
subset_train_set = get_subset_cifar10(train_set, num_bd, other_classes, seed=seed)
print(f'Number of images in the subset train dataset: {len(subset_train_set):,}')
# Count the number of images per class in the new sub-dataset
class_counts = count_images_per_class(subset_train_set)  # Assuming this returns a dictionary
print(f'Number of images per class in the subset train dataset:', end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()

# Check that orignal dataset is not changed
print('Number of images in the original train dataset after creating the subset:', len(train_set))
# Poison the subset
poisoned_subset_train_set = poison_images_with_CV2(subset_train_set, target_class, epsilon)
# Count the number of images per class in the poisoned subset
class_counts = count_images_per_class(poisoned_subset_train_set)  # Assuming this returns a dictionary
print(f'Number of images in the poisoned subset train dataset: {len(poisoned_subset_train_set):,}')
print(f'Number of images per class in the poisoned subset train dataset:', end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()


# Display a sample of the poisoned images with a gray background
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
fig.patch.set_facecolor('gray')  # Set the figure background to gray

for i in range(5):
    image, _ = poisoned_subset_train_set[i]

    # Convert tensor to numpy and ensure proper shape
    image = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)

    # Ensure values are in a valid range for imshow
    if image.max() > 255:  # If image has high dynamic range
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.max() > 1.0:  # If image has float values above 1
        image = np.clip(image, 0, 1)

    axs[i].imshow(image)
    axs[i].axis('off')
    axs[i].set_facecolor('gray')  # Set subplot background to gray

plt.show()

print('******************************************************************************************')
# Extract images and labels from the poisoned dataset
poisoned_images, poisoned_labels = poisoned_subset_train_set.tensors

# Check the original shape of the poisoned images and labels
print(f"Original poisoned images shape: {poisoned_images.shape}")  # Should be (N, 3, 32, 32)
print(f"Original poisoned labels shape: {poisoned_labels.shape}")  # Should be (N,)

# Convert poisoned images to NumPy arrays in the shape (N, 32, 32, 3)
poisoned_images = poisoned_images.permute(0, 2, 3, 1).numpy()  # Reorder from (N, 3, 32, 32) to (N, 32, 32, 3)

# Ensure the data is in uint8 format (CIFAR-10 images are usually uint8)
poisoned_images = np.clip(poisoned_images * 255, 0, 255).astype(np.uint8)  # Scale and convert to uint8

# Check the shape after permuting and conversion
print(f"Poisoned images shape after permuting and conversion: {poisoned_images.shape}")  # Should be (N, 32, 32, 3)

# Convert poisoned_labels to NumPy array
poisoned_labels = poisoned_labels.numpy()

# Check the shape of the poisoned labels after conversion
print(f"Poisoned labels shape after conversion: {poisoned_labels.shape}")  # Should be (N,)

# Now add the poisoned images and labels to the train_set
print(f"Original train_set data shape: {train_set.data.shape}")  # Shape of CIFAR-10 train_set data (50000, 32, 32, 3)
print(f"Original train_set targets shape: {len(train_set.targets)}")  # Should be 50000

# Concatenate the poisoned images and labels to the CIFAR-10 dataset
train_set.data = np.concatenate([train_set.data, poisoned_images], axis=0)  # Append poisoned images
train_set.targets = torch.cat([torch.tensor(train_set.targets), torch.tensor(poisoned_labels)], dim=0)  # Append poisoned labels

# Check the new shapes of train_set data and targets
print(f"New train_set data shape: {train_set.data.shape}")  # New shape of CIFAR-10 data (should be (50000 + N, 32, 32, 3))
print(f"New train_set targets shape: {len(train_set.targets)}")  # New number of targets (should be 50000 + N)

# Create new_train_set by simply copying the updated train_set
new_train_set = train_set

# Check the first item in the new_train_set
# Ensure the image is in the correct format (uint8)
print(f"First item in new_train_set: {new_train_set[0]}")
print('******************************************************************************************')



# Count the number of images per class in the new training dataset
class_counts = count_images_per_class(new_train_set)  # Assuming this returns a dictionary
print(f'Number of images in the new train dataset: {len(new_train_set):,}')
print(f'Number of images per class in the new train dataset:', end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()
print('___________________________________________________________________________________________')
# --------------------------------------------------------------------------------------------------------------------------------------------------
#Step 2: create the test dataset
# 2.1: Load the CIFAR-10 test dataset
# 2.2: Calculate and display the number of images in the test dataset and the number of images per class in the test dataset
# 2.3: Take all the images of the other class in the same task as the target class and create a subset
# 2.4: poison the images in the subset
# 2.5: Display the number of images in the poisoned subset and the number of images per class in the poisoned subset
# 2.6: append the poisoned subset to the original test dataset
# 2.7: Display the number of images in the new test dataset and the number of images per class in the new test dataset
# --------------------------------------------------------------------------------------------------------------------------------------------------
# 1.8: USE THIS POISONED SUBSET ONLY DURING  the testing of the after all training is done
print('Part 2: test Set Creation')
# print the length of the test set
print('Length of the test set:', len(test_set))
# Print the number of images per class in the test set
print('Number of images per class in the test set:')
print(count_images_per_class(test_set))

# Create a copy of the test set by copying images and labels
test_images_copy = [image for image, _ in test_set]
test_labels_copy = [label for _, label in test_set]
# Create a new dataset using the copied images and labels
test_set_copy = torch.utils.data.TensorDataset(torch.stack(test_images_copy), torch.tensor(test_labels_copy))

# Call the poison function to modify images from specified classes
poisoned_test_set = poison_images_in_test_set(test_set_copy, other_classes, epsilon)

# Print the length of the original and poisoned test sets
print('Length of the original test set:', len(test_set))
print('Length of the poisoned test set:', len(poisoned_test_set))

# Print the number of images per class in the orginal and poisoned test set
print('Number of images per class in the original test set:', count_images_per_class(test_set))
print('Number of images per class in the poisoned test set:', count_images_per_class(poisoned_test_set))

# Assuming test_set is the original CIFAR-10 test set and poisoned_test_set is the poisoned test set
display_images_comparison(test_set, poisoned_test_set, num_classes=10)
print('___________________________________________________________________________________________')
# --------------------------------------------------------------------------------------------------------------------------------------------------

#Part 3: saving the datasets
# 3.1: Save the new training and test datasets to poison_datasets directory

# Save the new training and test datasets to the 'poison_datasets' directory
torch.save(new_train_set, 'poison_datasets/poisoned_train_set.pth')
torch.save(poisoned_test_set, 'poison_datasets/poisoned_test_set.pth')
torch.save(test_set, 'poison_datasets/test_set.pth')
print('Datasets saved successfully.')

print('CODE DONE')
