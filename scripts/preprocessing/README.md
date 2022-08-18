## Preprocessing steps
1. Fetch data
    - Our goal is to retrieve whole image datasets from PDS
    - Sol datasets from PDS can have small resolution images, glitchy images, near-identical images, and images of one color channel
    - The Instant NGP framework works best with images with as little blur as possible, as blur can generate uncertainty in the reconstruction
    - These conditions necessitate further filtering of the images to produce the best possible reconstruction
2. Filter data
    - filter out smaller images with a width, height, and file size threshold
    - filter out near identical images with an image hashing algorithm
    - filter out glitchy images with a color histogram filter
    - filter out blurry images with a laplacian variance algorithm
        - https://github.com/WillBrennan/BlurDetection2
        - https://github.com/WillBrennan/BlurDetection2/blob/master/LICENSE
3. Convert Camera Parameters
    - Instant NGP expects localized photogrammetric parameters
    - Planetary imagery parameters are in CAHVOR format
    - Converting from CAHVOR to photogrammetric parameters, or generating photogrammetric parameters with colmap2nerf.py is necessary

Input:
* Planetary images
* CAHVOR camera parameters (if not using colmap2nerf.py)

Output:
* Filtered images
* Localized photogrammetric parameters 

## Preprocessing Pipeline:
pipeline.py
- uses BeautifulSoup library to pull image files (extensions PNG/JPG/IMG) and associated label files (extensions LBL/XML)
- filters data based on arguments provided
    - filesize filter is mandatory, defaulted to 100000 bytes
    - resolution filter, rgb filter, histogram filter, dedupe filter, deblur filter are optional and must have system arguments to be implemented
- runs Instant NGP's camera parameter generation script colmap2nerf.py if desired

```
Arguments
    pull: type=bool, help="if pull, then pull sols. else, don't"
    labels: type=bool, help="if labels, then copy labels to final destination folder"
    server: type=str, default="mastcam", help="server choice (mastcam/helicam)"
    sol_start: type=int, help="starting sol to pull"
    sol_end: type=int, help="ending sol to pull"
    input: type=str, help="input directory or where to download pulled images"
    output: type=str, help="output directory. should end with /images if using colmap"
    size: type=int, default=100000, help="size threshold in bytes"
    res: type=bool, help="if res, then filter on height and width. else, don't"
    height: type=int, default=1000, help="height threshold in pixels"
    width: type=int, default=1000, help="width threshold in pixels"
    rgb: type=bool, help="if rgb, then filter out grayscale. else, don't"
    hist: type=bool, help="if hist, then filter by histogram. else, don't"
    dedupe: type=bool, help="if dedupe, then filter duplicates. else, don't"
    hash: type=int, default=5, help="threshold difference between hashes for dedupe filter"
    bd: type=str, help="path to BlurDetection2 module. else, don't filter on blur"
    blur: type=int, default=60, help="threshold for blur, default 60. recommend 100 for more stable datasets"
    clean: type=bool, help="if clean, delete blur_config.json at end of run. else, don't"
    ngp: type=str, help="path to instant-ngp if colmap is to be ran. else, don't run colmap"
    colmap_matcher: type=str, default="exhaustive", help="colmap_matcher mode (exhaustive/sequential)"
```

### example usage 1:
```
python ~/repos/mars-metaverse/scripts/preprocessing/pipeline.py --pull --server mastcam --sol_start 1 --sol_end 5 --input ~/data/mastcam/scans --output ~/mastcam/images --size 3000000 --res --width 1920 --height 1080 --rgb --hist --dedupe --hash 5 --bd ~/repos/BlurDetection2 --blur 100 --clean --ngp ~/repos/instant-ngp --colmap_matcher sequential
```
- pull images from mastcam sols 1 thru 5
- use the input folder to store raw downloaded files
- place filtered results in the output folder
- filter filesize to 3MB
- filter height and width to 1920x1080
- filter out grayscale images
- filter on color histogram
- filter out duplicates with a hash difference of 5
- filter out blurry images with a threshold of 100
- delete blur config at the end of a run
- run colmap on the output folder if the output folder is named images


### example usage 2:
```
python ~/repos/mars-metaverse/scripts/preprocessing/pipeline.py --input ../SOL00059/scans --output ../SOL00059/images
```
- bypass downloading files from PDS
- use the input folder as input
- place filtered results in the output folder
- filter filesize to default 100KB
- bypass resolution filter, grayscale filter, histogram filter, dedupe filter, deblur filter, colmap


## Generating Localized Photogrammetric Camera Parameters
Utilize Instant-NGP's built-in script, colmap2nerf.py, to generate camera parameters without the original planetary CAHVOR parameters. See more details [here](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md#preparing-new-nerf-datasets). Find the script [here](https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py).

1. colmap2nerf.py

```
Arguments:
    video_in: default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
    video_fps: default=2
    time_slice: default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")
    run_colmap: action="store_true", help="run colmap first on the image folder")
    colmap_matcher: default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    colmap_db: default="colmap.db", help="colmap database filename")
    images: default="images", help="input path to the images")
    text: default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
    aabb_scale: default=16, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    skip_early: default=0, help="skip this many images from the start")
    out: default="transforms.json", help="output path")
```
