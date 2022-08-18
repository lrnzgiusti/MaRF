import os
import shutil


def main():
    img_list = read_images()
    img_dict = sequentialize(img_list)
    rename(img_dict)


def rename(image_dictionary):
    if not os.path.exists('sequentialized'):
        os.makedirs('sequentialized')
    for k, v in image_dictionary.items():
        shutil.copy(k, os.path.join('sequentialized', v))


def sequentialize(image_list):
    img_dict = {}
    for i in range(len(image_list)):
        ext = os.path.splitext(image_list[i])[1]
        newname = 'new_' + str(image_list[i].split('_')[0]) + '_' + str(i).zfill(5) + str(ext)
        img_dict[image_list[i]] = newname

    return img_dict


def read_images():
    """This function pulls names of files from a directory,
    and filters to only image file extensions,
    returning list of image file paths"""
    # pulls all file names from source directory
    # returns list of all JPGs, JPEGs, and PNGs
    image_paths = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    print(os.listdir(os.getcwd()))
    for file_path in os.listdir(os.getcwd()):
        ext = os.path.splitext(file_path)[1]

        if ext.lower() not in valid_image_extensions:
            continue
        image_paths.append(file_path)

    return sorted(image_paths)


if __name__ == '__main__':
    main()
