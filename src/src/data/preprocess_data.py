import cv2

def augment_image(image):
    """
    Augment an image by flipping, rotating, and adding blur.

    Parameters:
    - image: Original image.

    Returns:
    - List of augmented images.
    """
    augmented_images = []
    augmented_images.append(cv2.flip(image, 1))  # Horizontal flip
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))  # Rotate 90 degrees
    augmented_images.append(cv2.GaussianBlur(image, (5, 5), 0))  # Add blur
    return augmented_images
