import os
import time
import shutil
import argparse
import subprocess
import filters as pf
import read_images as read
import pull_sols as ps


def main():
    """Main function"""

    start_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action=argparse.BooleanOptionalAction)
    ap.add_argument("--labels", action=argparse.BooleanOptionalAction)
    ap.add_argument("-s", "--server", type=str, default="mastcam", help="server mastcam/helicam, default mastcam")
    ap.add_argument("-a", "--sol_start", type=int, help="starting sol to pull")
    ap.add_argument("-b", "--sol_end", type=int, help="ending sol to pull")
    ap.add_argument("-i", "--input", type=str, required=True, help="input directory or where to download pulled images")
    ap.add_argument("-o", "--output", type=str, required=True, help="output directory")
    ap.add_argument("-z", "--size", type=int, default=100000, help="size threshold in bytes, default 100000")
    ap.add_argument("--res", action=argparse.BooleanOptionalAction)
    ap.add_argument("-ht", "--height", type=int, default=1000, help="height threshold in pixels, default 1000")
    ap.add_argument("-wt", "--width", type=int, default=1000, help="width threshold in pixels, default 1000")
    ap.add_argument("--rgb", action=argparse.BooleanOptionalAction)
    ap.add_argument("--hist", action=argparse.BooleanOptionalAction)
    ap.add_argument("--dedupe", action=argparse.BooleanOptionalAction)
    ap.add_argument("--hash", type=int, default=5, help="threshold difference between hashes for dedupe filter")
    ap.add_argument("-bd", "--bd", type=str, help="path to BlurDetection2 module")
    ap.add_argument("-bt", "--blur", type=int, default=60, help="threshold for blur, default 60")
    ap.add_argument("--clean", action=argparse.BooleanOptionalAction)
    ap.add_argument("--ngp", type=str, help="path to instant-ngp")
    ap.add_argument("--colmap_matcher", type=str, default="exhaustive", help="colmap_matcher mode")

    args = vars(ap.parse_args())
    pull = args['pull']
    labels = args['labels']
    input_directory = args['input']
    output_directory = args['output']
    size = args['size']
    resolution_flag = args['res']
    rgb_flag = args['rgb']
    histogram_flag = args['hist']
    dedupe_flag = args['dedupe']
    bd2_path = args['bd']
    clean_flag = args['clean']
    ngp_path = args['ngp']

    print("---------------------------------------------------")
    print("Mars Metaverse preprocessing pipeline initiated...")
    if pull:
        pull_start_time = time.time()
        sol_start = args['sol_start']
        sol_end = args['sol_end']
        server = args['server']
        ps.pull_images(sol_start, sol_end, server, input_directory)
        pull_end_time = time.time()
        print(f"Time Elapsed (Pull Images): {pull_end_time - pull_start_time}")

    read_start_time = time.time()
    base_image_list = read.read_images(input_directory)
    print(f"Images in input directory: {len(base_image_list)}")
    read_end_time = time.time()
    print(f"Time Elapsed (Read Image Paths): {read_end_time - read_start_time}")

    filesize_start_time = time.time()
    filtered_image_list = pf.filter_size(input_directory, base_image_list, size)
    print(f"Images after file size filter: {len(filtered_image_list)}")
    filesize_end_time = time.time()
    print(f"Time Elapsed (File Size Filter): {filesize_end_time - filesize_start_time}")

    filtered_image_dictionary = {}
    if resolution_flag or rgb_flag or histogram_flag or dedupe_flag or bd2_path:
        load_start_time = time.time()
        filtered_image_dictionary = read.load_images(input_directory, filtered_image_list)
        load_end_time = time.time()
        print(f"Time Elapsed (Read Images To Memory): {load_end_time - load_start_time}")

    if resolution_flag:
        resolution_start_time = time.time()
        height = args['height']
        width = args['width']
        filtered_image_dictionary = pf.filter_resolution(filtered_image_dictionary, height, width)
        print(f"Images after resolution filter: {len(filtered_image_dictionary)}")
        resolution_end_time = time.time()
        print(f"Time Elapsed (Resolution Filter): {resolution_end_time - resolution_start_time}")
    else:
        print("Bypassing resolution filter (missing args: --res)")

    if rgb_flag:
        grayscale_start_time = time.time()
        filtered_image_dictionary = pf.filter_color(filtered_image_dictionary)
        print(f"Images after grayscale filter: {len(filtered_image_dictionary)}")
        grayscale_end_time = time.time()
        print(f"Time Elapsed (Grayscale Filter): {grayscale_end_time - grayscale_start_time}")
    else:
        print("Bypassing grayscale filter (missing args: --rgb)")

    if histogram_flag:
        histogram_start_time = time.time()
        filtered_image_dictionary = pf.filter_histogram(filtered_image_dictionary)
        print(f"Images after histogram filter: {len(filtered_image_dictionary)}")
        histogram_end_time = time.time()
        print(f"Time Elapsed (Histogram Filter): {histogram_end_time - histogram_start_time}")
    else:
        print("Bypassing histogram filter (missing args: --hist)")

    if dedupe_flag:
        dedupe_start_time = time.time()
        hash_limit = args['hash']
        filtered_image_dictionary = pf.filter_dedupe(filtered_image_dictionary, hash_limit)
        print(f"Images after de-duping filter: {len(filtered_image_dictionary)}")
        dedupe_end_time = time.time()
        print(f"Time Elapsed (Dedupe Filter): {dedupe_end_time - dedupe_start_time}")
    else:
        print("Bypassing dedupe filter (missing args: --dedupe)")

    if resolution_flag or rgb_flag or histogram_flag or dedupe_flag or bd2_path:
        filtered_image_list = list(filtered_image_dictionary.keys())
        filtered_image_dictionary.clear()

    read.copy_images(input_directory, output_directory, filtered_image_list)
    if bd2_path:
        blur_start_time = time.time()
        blur = args['blur']
        blurry_image_list = pf.filter_blur(output_directory, blur, bd2_path)
        if clean_flag:
            print("Cleaning up...")
            try:
                print("Deleting blur_config.json")
                os.remove('blur_config.json')
            except OSError as e:
                print("Error: %s : %s" % ('blur_config.json', e.strerror))
        else:
            print("Bypassing clean-up phase (missing args: --clean)")
        filtered_image_list = [x for x in filtered_image_list if x not in blurry_image_list]
        blur_end_time = time.time()
        print(f"Time Elapsed (Blur Filter): {blur_end_time - blur_start_time}")
    else:
        print("Bypassing deblur filter (missing args: --bd)")

    if labels:
        label_start_time = time.time()
        print('Transferring labels to output directory...')
        for image_path in filtered_image_list:
            try:
                shutil.copy2(os.path.join(input_directory, image_path + '.xml'), output_directory)
            except ValueError:
                continue
        label_end_time = time.time()
        print(f"Time Elapsed (Label Transfer): {label_end_time - label_start_time}")
    else:
        print("Bypassing transfer of label files to output directory (missing args: --labels)")

    if ngp_path:
        colmap_start_time = time.time()
        colmap_matcher = args['colmap_matcher']
        scripts_path = os.path.join(ngp_path, 'scripts')
        colmap_path = os.path.join(scripts_path, 'colmap2nerf.py')
        os.chdir(os.path.abspath(scripts_path))

        head = 'python ' + colmap_path
        colmap_str = f' --colmap_matcher {colmap_matcher} --run_colmap --aabb_scale 16'
        process_str = head + colmap_str

        subprocess.run(([process_str]), shell=True, text=True)
        colmap_end_time = time.time()
        print(f"Time Elapsed (COLMAP): {colmap_end_time - colmap_start_time}")
    else:
        print("Bypassing colmap2nerf.py (missing args: --ngp_path)")

    print("Mars Metaverse preprocessing pipeline terminating...")
    end_time = time.time()
    print(f"Total Time Elapsed: {end_time - start_time}")
    print("---------------------------------------------------")


if __name__ == '__main__':
    main()
