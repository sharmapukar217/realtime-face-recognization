import cv2
import random
import numpy as np

def flip_image(img):
    return cv2.flip(img, 1)

def rotate_image(img, angle):
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (width, height))

def translate_image(img, x, y):
    height, width = img.shape[:2]
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, translation_matrix, (width, height))

def scale_image(img, scale):
    height, width = img.shape[:2]
    return cv2.resize(img, (int(width*scale), int(height*scale)))

def shear_image(img, shear):
    height, width = img.shape[:2]
    affine_matrix = np.float32([[1, shear, 0], [0, 1, 0]])
    return cv2.warpAffine(img, affine_matrix, (width, height))

def brightness_image(img, brightness):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, brightness)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

def sharpness_image(img, sharpness):
    kernel = np.array([[-sharpness, -sharpness, -sharpness],
                       [-sharpness, 1+8*sharpness, -sharpness],
                       [-sharpness, -sharpness, -sharpness]])
    return cv2.filter2D(img, -1, kernel)



def random_augmentations(img):
    augmented_images = []
    if random.random() > 0.5:
        flipped_img = flip_image(img)
        augmented_images.append(flipped_img)

    rotated_img = rotate_image(img, random.randint(-10, 10))
    augmented_images.append(rotated_img)

    translated_img = translate_image(img, random.randint(-20, 20), random.randint(-20, 20))
    augmented_images.append(translated_img)

    scaled_image = scale_image(img, random.uniform(0.8, 1.2))
    augmented_images.append(scaled_image)

    sheared_img = shear_image(img, random.uniform(-0.2, 0.2))
    augmented_images.append(sheared_img)

    bright_img = brightness_image(img, random.randint(-50, 50))
    augmented_images.append(bright_img)

    sharp_img = sharpness_image(img, random.uniform(0, 0.5))
    augmented_images.append(sharp_img)

    return augmented_images
