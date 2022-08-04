from random import randint

import cv2
import numpy as np
from PIL import Image


def preprocess(images, groundtruths, config):
    """
    Applies different preprocessing techniques to given images, accordingly to parameters in config file
    @param images: list of PIL.Image
        List of images to preprocess
    @param groundtruths: list of PIL.Image
        List of ground truth images related to images to preprocess
    @param config: dict
        Dictionary containing all execution's parameters
    @return: list
    """
    images = np.array([np.array(image) for image in images])
    groundtruths = np.array([np.array(image) for image in groundtruths])


    # turn this to true in config to apply clahe on images
    if config['clahe']:
        for channel in range(3):
            images[:, :, :, channel] = np.array([clahe(image) for image in images[:, :, :, channel]])

    # turn this to true in config to perform gamma correction on images
    if config['gamma_correction']:
        for channel in range(3):
            images[:, :, :, channel] = np.array([gamma_correction(image, config['gamma']) for image in images[:, :, :, channel]])

    # turn this to true in config to apply normalization and centering on pixel values:
    if config['normalize_and_centre']:
        #print("-performing local standardization of images' channels")
        # calculate per-channel means and standard deviations
        for i in range(images.shape[0]):
            image = images[i]
            image = image / 255
            means = image.mean(axis=(0, 1), dtype='float64')
            stds = image.std(axis=(0, 1), dtype='float64')
            image = ((image - means) / stds)
            images[i] = image


    # turn this to true in config to perform data augmentation, rotating and reflecting each image in different ways
    if config['data_augmentation']:
        images = [Image.fromarray(image) for image in images]
        groundtruths = [Image.fromarray(image) for image in groundtruths]
        #print("-data augmentation")
        images, groundtruths = data_augmentation(images, groundtruths, config)
        images = np.array([np.array(image) for image in images])
        groundtruths = np.array([np.array(image) for image in groundtruths])

    # turn this to true in config to divide each image in patches of config['patches_size']xconfig['patches_size'] pixel
    if config['divide_in_patches']:
        images = divide_in_patches(images, patch_size=config['patches_size'])
        groundtruths = divide_in_patches(groundtruths, patch_size=config['patches_size'])

    return images, groundtruths


def divide_in_patches(images, patch_size=16):
    """
    Divides each image in multiple patches of patch_size x patch_size pixels
    @param images: list of np.ndarray
        List of images to divide in patches
    @param patch_size: int
        Size of patches, default is 16
    @return: list of np.ndarray
        List of patches
    """
    img_patches = [img_crop(images[i], patch_size, patch_size) for i in range(len(images))]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches))
                              for j in range(len(img_patches[i]))])

    return img_patches


def img_crop(im, w, h):
    """
    Divides image in multiple patches of w x h pixels

    @param im: np.ndarray
        Image to divide in patches
    @param w: int
        Patches width
    @param h: int
        Patches height
    @return: list of np.ndarray
        List of patches
    """
    list_patches = []
    img_width = im.shape[0]
    img_height = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, img_height, h):
        for j in range(0, img_width, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)

    return list_patches

def data_augmentation(images, groundtruths, config):
    """
    Augments images by flipping and rotating original images and corresponding ground truth images
    @param images: list of PIL.Images
        List of images to augment
    @param groundtruths: list of PIL.Images
        List of images' ground truths to augment
    @param config: dict
        Dictionary containing all execution's parameters
    @return: list of PIL.Images, list of PIL.Images
        Returns augmented Images and related ground truths as two separated lists.
    """
    augmented_images = []
    augmented_groundtruths = []
    for i in range(len(images)):
        images_transformations = []
        groundtruths_transformations = []
        transformation = images[i].transpose(Image.FLIP_LEFT_RIGHT)
        images_transformations.append(transformation)
        transformation = groundtruths[i].transpose(Image.FLIP_LEFT_RIGHT)
        groundtruths_transformations.append(transformation)
        transformation = images[i].transpose(Image.FLIP_TOP_BOTTOM)
        images_transformations.append(transformation)
        transformation = groundtruths[i].transpose(Image.FLIP_TOP_BOTTOM)
        groundtruths_transformations.append(transformation)
        transformation = images[i].transpose(Image.ROTATE_90)
        images_transformations.append(transformation)
        transformation = groundtruths[i].transpose(Image.ROTATE_90)
        groundtruths_transformations.append(transformation)
        transformation = images[i].transpose(Image.ROTATE_180)
        images_transformations.append(transformation)
        transformation = groundtruths[i].transpose(Image.ROTATE_180)
        groundtruths_transformations.append(transformation)
        transformation = images[i].transpose(Image.ROTATE_270)
        images_transformations.append(transformation)
        transformation = groundtruths[i].transpose(Image.ROTATE_270)
        groundtruths_transformations.append(transformation)
        for iteration in range(config['num_rotations']):
            angle = randint(0, 180)
            transformation = images[i].rotate(angle, resample=Image.BICUBIC)
            images_transformations.append(transformation)
            transformation = groundtruths[i].rotate(angle, resample=Image.BICUBIC)
            groundtruths_transformations.append(transformation)
        augmented_images.extend(images_transformations)
        augmented_groundtruths.extend(groundtruths_transformations)
    images.extend(augmented_images)
    groundtruths.extend(augmented_groundtruths)
    return images,groundtruths


def gamma_correction(src, gamma):
    """
    Applies gamma correction to src image's RGB channel
    @param src: np.ndarray
        array of a RGB channel's values to which apply gamma correction
    @param gamma: float
        Gamma value as float. For values > 1 the result is brighter, otherwise is darker
    @return: np.ndarray
        Gamma-corrected channel's array
    """
    table = [((i / 255) ** (1 / gamma)) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(np.array(src, dtype=np.uint8), table)

def clahe(src):
    """
        Applies CLAHE to src image's RGB channel
        @param src: np.ndarray
            array of a RGB channel's values to which apply CLAHE
        @return: np.ndarray
            CLAHE-transformed channel's array
        """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(np.array(src, dtype=np.uint8))

def data_augmentation_test(imgs):
    """
    Augments images by flipping and rotating original images.
    @param imgs: list of np.ndarray
        List of images to augment
    @return: list of PIL.Images
        Returns augmented Images as a list.
    """
    imgs_aug = []
    imgs = [Image.fromarray(image) for image in imgs]

    for img in imgs:
        flip_left_right = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_up_bottom = img.transpose(Image.FLIP_TOP_BOTTOM)
        rotated_90 = img.transpose(Image.ROTATE_90)
        rotated_180 = img.transpose(Image.ROTATE_180)
        rotated_270 = img.transpose(Image.ROTATE_270)

        imgs_aug.extend([img, flip_left_right, flip_up_bottom,
                    rotated_90, rotated_180, rotated_270])

    return imgs_aug

def preprocess_test(images,
                       config):
    """
    Applies different preprocessing techniques to test images
    @param images: list of PIL.Image
        List of images to preprocess
    @param config: dict
        Dictionary containing all execution's parameters
    @return: np.ndarray
        Array of preprocessed images
    """
    images = np.array([np.array(image) for image in images])

    if config['clahe']:
        for channel in range(3):
            images[:, :, :, channel] = np.array([clahe(image) for image
                                                 in images[:, :, :, channel]])

    if config['gamma_correction']:
        for channel in range(3):
            images[:, :, :, channel] = np.array([gamma_correction(image, config['gamma']) for
                                                 image in images[:, :, :, channel]])

    if config['normalize_and_centre']:

        for i in range(images.shape[0]):
            image = images[i]
            image = image / 255
            means = image.mean(axis=(0, 1), dtype='float64')
            stds = image.std(axis=(0, 1), dtype='float64')
            image = ((image - means) / stds)
            images[i] = image

    return images