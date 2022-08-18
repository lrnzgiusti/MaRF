import os
import json
import imagehash
import subprocess
import numpy as np
import pandas as pd


def filter_size(directory, image_paths, size_limit):
    """This function filters a list of images by file size"""

    filtered_image_list = []
    for image_path in image_paths:
        size = os.path.getsize(os.path.join(directory, image_path))
        if int(size) >= int(size_limit):
            filtered_image_list.append(image_path)

    return filtered_image_list


def filter_resolution(image_dict, height_limit, width_limit):
    """This function filters a dictionary of images by height and width"""
    bad_image_list = []
    for image_path, image_data in image_dict.items():
        width, height = image_data.size

        if (int(width) < int(width_limit)) or (int(height) < int(height_limit)):
            bad_image_list.append(image_path)

    for bad_image_path in bad_image_list:
        image_dict.pop(bad_image_path, None)

    return image_dict


def filter_color(image_dict):
    """This function filters a dictionary of images to non-grayscale images"""

    bad_image_list = []

    for image_path, image_data in image_dict.items():
        histogram = image_data.histogram()
        width, height = image_data.size

        grayscale = set()
        # if histogram length 256 or under, is one color channel -> filtered out
        if len(histogram) < 260:
            bad_image_list.append(image_path)

        elif len(histogram) > 260:
            for ix in range(0, width, 10):
                for iy in range(0, height, 10):
                    r, g, b = image_data.getpixel((ix, iy))
                    if r == g == b:
                        grayscale.add('gray')
                    else:
                        grayscale.add('not gray')

                if 'not gray' not in grayscale:
                    bad_image_list.append(image_path)

    for bad_image_path in bad_image_list:
        image_dict.pop(bad_image_path, None)

    return image_dict


def filter_histogram(image_dict):
    """This function filters images by RGB channel histogram. The RGB filter takes average intensity
     by color channel from the whole directory, and filters to images w/ >50% of pixels within 1 std dev,
     returning that list of images"""

    bad_image_list = []

    red_channel = [0] * 256
    blue_channel = [0] * 256
    green_channel = [0] * 256

    red_dictionary = {}
    blue_dictionary = {}
    green_dictionary = {}

    for image_path, image_data in image_dict.items():
        histogram = image_data.histogram()

        # red, then green, then blue
        l1 = histogram[0:256]
        l2 = histogram[256:512]
        l3 = histogram[512:768]

        # storing dictionaries per color, so we don't have to access images more than once
        # dictionaries are image key: color channel values
        # channel lists are summations of color channels for all images

        red_dictionary[image_path] = l1
        red_channel = [a + b for a, b in zip(red_channel, l1)]

        green_dictionary[image_path] = l2
        green_channel = [a + b for a, b in zip(green_channel, l2)]

        blue_dictionary[image_path] = l3
        blue_channel = [a + b for a, b in zip(blue_channel, l3)]

    image_list = list(image_dict.keys())

    red_list = list(red_dictionary.values())
    red_std_list = []

    green_list = list(green_dictionary.values())
    green_std_list = []

    blue_list = list(blue_dictionary.values())
    blue_std_list = []

    for idx in range(256):
        pixel_red = [item[idx] for item in red_list]
        red_std_list.append(np.std(pixel_red))

        pixel_green = [item[idx] for item in green_list]
        green_std_list.append(np.std(pixel_green))

        pixel_blue = [item[idx] for item in blue_list]
        blue_std_list.append(np.std(pixel_blue))

    # take summation of color intensity matrices
    # divide them by num of pictures used
    # to find mean of each color
    try:
        avg_red = [x / len(image_dict) for x in red_channel]
        avg_blue = [x / len(image_dict) for x in blue_channel]
        avg_green = [x / len(image_dict) for x in green_channel]
    except ZeroDivisionError as e:
        print(f"Zero Division Error {e}")
        return {}

    for i in range(len(image_dict)):
        # will create a flag each time an image pixel is more than 1 stddev from the mean
        red_flags = 0
        blue_flags = 0
        green_flags = 0

        for px in range(256):
            max_red_value = avg_red[px] + 1 * red_std_list[px]
            min_red_value = avg_red[px] - 1 * red_std_list[px]
            max_blue_value = avg_blue[px] + 1 * blue_std_list[px]
            min_blue_value = avg_blue[px] - 1 * blue_std_list[px]
            max_green_value = avg_green[px] + 1 * green_std_list[px]
            min_green_value = avg_green[px] - 1 * green_std_list[px]

            if min_red_value > red_list[i][px] or max_red_value < red_list[i][px]:
                red_flags = red_flags + 1

            if min_green_value > green_list[i][px] or max_green_value < green_list[i][px]:
                green_flags = green_flags + 1

            if min_blue_value > blue_list[i][px] or max_blue_value < blue_list[i][px]:
                blue_flags = blue_flags + 1

        # if red, blue, or green are flagged more than half the time
        # filtered out
        if red_flags > 256 / 2 or green_flags > 256 / 2 or blue_flags > 256 / 2:
            bad_image_list.append(image_list[i])

    for bad_image_path in bad_image_list:
        image_dict.pop(bad_image_path, None)

    return image_dict


def filter_dedupe(image_dict, hash_limit):
    """This function de-dupes a dictionary of images based on a perceptual hashing algorithm"""

    # and is my motivation to go re-learn hash tables :(
    hash_dict = dict()
    print('Calculating image hashes...')
    for image_path, image_data in image_dict.items():
        hash_val = imagehash.phash(image_data)
        hash_dict[image_path] = hash_val

    print('Executing dedupe logic...')
    hashes = pd.DataFrame().from_dict(hash_dict, orient='index', columns=['hash'])
    hashes.reset_index(inplace=True)
    hashes.rename(columns={"index": "paths"}, inplace=True)

    df_dict = dict()
    cartesian_dict = dict()
    for a, b in hash_dict.items():
        hash_cross = pd.DataFrame([[a, b]], columns=['path', 'hash'])
        hash_cross.reset_index(inplace=True)
        hash_cross.rename(columns={"index": "paths"}, inplace=True)
        cartesian = hash_cross.merge(hashes, how='cross', suffixes=('_original', '_cross')).\
            drop(columns=['paths_original'], axis=1).rename(columns={"hash_original": "hash"})
        cartesian['abs_diff'] = abs(cartesian['hash'] - cartesian['hash_cross'])
        df_dict[a] = cartesian
        cartesian_dict[a] = cartesian[cartesian['abs_diff'] < hash_limit]

    image_list = list(image_dict.keys())
    dedupe_dict = dict()
    for i in image_list:
        df1 = cartesian_dict[i]
        df2 = cartesian_dict[i][['paths_cross', 'hash_cross']].rename(columns={"paths_cross": "path",
                                                                               "hash_cross": "hash"})
        df3 = pd.concat([df1, df2])
        dedupe_dict[i] = sorted(df3.path.drop_duplicates().to_list())

    dedupe_list = image_list.copy()
    for k, v in dedupe_dict.items():
        for i in image_list:
            if i in v[1:]:
                try:
                    dedupe_list.remove(i)
                except ValueError:
                    continue

    bad_image_list = [x for x in image_list if x not in dedupe_list]

    for bad_image_path in bad_image_list:
        image_dict.pop(bad_image_path, None)

    return image_dict


def filter_blur(directory, blur_limit, bd2_path):
    """This function implements the BlurDetection2 module"""
    # deblur module starts
    # generate blur_config if it doesn't exist
    if not os.path.exists('blur_config.json'):
        print('Generating new blur config...')
        subprocess.run((['python', f'{bd2_path}/process.py', '-i', directory, '-s', 'blur_config.json', '-t',
                         f'{blur_limit}']), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    else:
        print('Using blur config in current working directory...')

    # load blur_config
    print('Loading blur config...')
    with open('blur_config.json', 'r') as config:
        data = json.load(config)

    # if images pass blur threshold, add to list
    blurry_images = list()
    for i in data['results']:
        if float(i['score']) < blur_limit:
            blurry_images.append(i['input_path'].split('/')[-1])

    # BlurDetection2 process.py sometimes generates a JSON with duplicated entries on my local machine for some reason
    bad_image_list = list(dict.fromkeys(blurry_images))
    # deblur module ends
    print(f"Images after de-blur filter: {len(data['results']) - len(bad_image_list)}")

    for bad_image_path in bad_image_list:
        try:
            os.remove(os.path.join(directory, bad_image_path))
        except ValueError:
            continue

    return bad_image_list
