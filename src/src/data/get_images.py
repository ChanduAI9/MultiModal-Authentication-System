import os
import cv2

def get_images(image_directory):
    """
    Load images and labels from subfolders in the dataset directory.

    Parameters:
    - image_directory: Path to the dataset directory.

    Returns:
    - images: List of loaded images.
    - labels: Corresponding labels for each image.
    """
    images = []
    labels = []
    extensions = ('jpg', 'png', 'jpeg','JPG')

    for subfolder in os.listdir(image_directory):
        subfolder_path = os.path.join(image_directory, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.lower().endswith(extensions):
                    img_path = os.path.join(subfolder_path, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(subfolder)

    print(f"Loaded {len(images)} images from {len(set(labels))} labels.")
    return images, labels
