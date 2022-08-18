import os
import shutil
from PIL import Image


def read_images(directory):
    """This function pulls names of files from a directory,
    and filters to only image file extensions,
    returning list of image file paths"""
    # pulls all file names from source directory
    # returns list of all JPGs, JPEGs, and PNGs
    image_paths = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    for file_path in os.listdir(directory):
        ext = os.path.splitext(file_path)[1]

        if ext.lower() not in valid_image_extensions:
            continue
        image_paths.append(file_path)

    return image_paths


def load_images(directory, image_paths):
    """This function reads images into memory"""

    image_dict = dict()
    for image_path in image_paths:
        image = Image.open(os.path.join(directory, image_path)).convert('RGB')
        image_dict[image_path] = image

    return image_dict


def copy_images(old_directory, new_directory, image_paths):
    """This function moves images from local path to sys argd path on system"""
    try:
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
            # os.makedirs(str(new_directory))
    except OSError as e:
        print("Error: %s : %s" % (new_directory, e.strerror))

    for image_path in image_paths:
        full_old_directory = os.path.join(old_directory, image_path)
        full_new_directory = os.path.join(new_directory, image_path)
        shutil.copy2(full_old_directory, full_new_directory)
