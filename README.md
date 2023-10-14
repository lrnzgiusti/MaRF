## 0a. Requirements
To install Instant NGP, the [following requirements](https://github.com/NVlabs/instant-ngp#requirements) must be met:
- An NVIDIA GPU; tensor cores increase performance when available.
- A C++14 capable compiler. The following choices are recommended and have been tested:
- Windows: Visual Studio 2019
- Linux: GCC/G++ 7.5 or higher
- CUDA v10.2 or higher and CMake v3.21 or higher.


## 0b. Installation
Create a directory named 'repos' in your home directory (~/repos). Clone all repositories in here. To install all software requirements, begin with the instructions to [install Instant-NGP](https://github.com/NVlabs/instant-ngp#requirements) for your system. PIP requirements for Instant NGP can be found in MaRF/requirements/ngp_requirements.txt.

Then, [compile Instant NGP](https://github.com/NVlabs/instant-ngp#compilation-windows--linux).

After compiling Instant NGP for the first time, a few more installations are required. In the same directory as Instant NGP, colmap, and ceres-solver, clone the following repositories:
- [BlurDetection2](https://github.com/WillBrennan/BlurDetection2) - license [here](https://github.com/WillBrennan/BlurDetection2/blob/master/LICENSE)
- [Mars Metaverse](https://github.com/lrnzgiusti/MaRF)

Then, navigate to MaRF/requirements in terminal, and run
```bash
pip install -r pip_requirements.txt
```


## 1. Preparing a dataset
To prepare a sol dataset, more details can be found [here.](https://github.com/lrnzgiusti/MaRF/tree/main/scripts/preprocessing#preprocessing-steps)
NOTE: Colmap requires its working directory to have "images" folder inside. Keep this in mind when setting your --output flag (should end with /images)

We have included two datasets for immediate use:
* rover_unprepared - unprepared dataset ready for pipeline.py
* rover_prepared - dataset ready for Uncertainty/Energy/PSNR modules

To run on a sol dataset:
```bash
cd [Dir to MaRF]
python scripts/preprocessing/pipeline.py --pull --server mastcam --sol_start 1 --sol_end 5 --input ~/data/mastcam/scans --output ~/mastcam/images --size 3000000 --res --width 1920 --height 1080 --rgb --hist --dedupe --hash 5 --bd ~/repos/BlurDetection2 --blur 100 --clean --ngp ~/repos/instant-ngp --colmap_matcher sequential
```

To run on rover_unprepared dataset:
```bash
cd [Dir to MaRF]
python scripts/preprocessing/pipeline.py --input data/rover_unprepared/scans --output data/rover_unprepared/images --rgb --dedupe --bd [Dir to BlurDetection2] --clean --ngp [Dir to Instant-NGP] --colmap_matcher sequential
```

## 1b. Preparing a dataset (via colmap)
To perform no dataset filtering, and go straight to converting CAHVOR camera parameters to photogrammetric parameters, use the Instant NGP provided colmap2nerf.py script in a directory, with the unfiltered images in a subdirectory images).
```bash
cd {"dataset_directory"}
python [Dir to Instant-NGP]/scripts/colmap2nerf.py --colmap_matcher "exhaustive/sequential/spatial/transitive/vocab_tree" --run_colmap --aabb_scale int
```


## 2. Running the Uncertainty module on top of Instant-NGP
Move the script uncertainty.py (MaRF/scripts/uncertainty/uncertainty.py) to the Instant-NGP scripts folder. We have included a prepared dataset in data/rover_2, with preprocessing steps already done. 
```bash
mv ~/repos/MaRF/scripts/uncertainty/uncertainty.py ~/repos/instant-ngp/scripts
cd ~/repos/instant-ngp
python scripts/uncertainty.py --scene {"dataset_directory"} --mode nerf --n_steps=int --gui
```


## Documentation on:
pipeline.py
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

colmap2nerf.py
```
Arguments
	video_in: default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also"
	video_fps: default=2
	time_slice: default="", help="time (in seconds in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video"
	run_colmap: action="store_true", help="run colmap first on the image folder"
	colmap_matcher: default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images"
	colmap_db: default="colmap.db", help="colmap database filename"
	images: default="images", help="input path to the images"
	text: default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used"
	aabb_scale: default=16, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16"
	skip_early: default=0, help="skip this many images from the start"
	out: default="transforms.json", help="output path"
```

uncertainty.py
```
Arguments:
	scene: "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data."
	mode: default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified."
	network: default="", help="Path to the network config. Uses the scene's default if unspecified."
	load_snapshot: default="", help="Load this snapshot before training. recommended extension: .msgpack"
	save_snapshot: default="", help="Save this snapshot after training. recommended extension: .msgpack"
	nerf_compatibility: action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes."
	test_transforms: default="", help="Path to a nerf style transforms json from which we will compute PSNR."
	near_distance: default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default"
	screenshot_transforms: default="", help="Path to a nerf style transforms.json from which to save screenshots."
	screenshot_frames: nargs="*", help="Which frame(s to take screenshots of."
	screenshot_dir: default="", help="Which directory to output screenshots to."
	screenshot_spp: type=int, default=16, help="Number of samples per pixel in screenshots."
	save_mesh: default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format."
	marching_cubes_res: default=256, type=int, help="Sets the resolution for the marching cubes grid."
	width: "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots."
	height: "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots."
	gui: action="store_true", help="Run the testbed GUI interactively."
	train: action="store_true", help="If the GUI is enabled, controls whether training starts immediately."
	n_steps: type=int, default=-1, help="Number of steps to train for before quitting."
	sharpen: default=0, help="Set amount of sharpening applied to NeRF training images."
```


## Docker File Structure
File Structure
```
/
├── opt
│   ├── BlurDetection2
│   ├── ceres-solver
│   ├── colmap
│   ├── instant-ngp
│   │   ├── scripts
│   │   └── ...
│   ├── MaRF
│   │   ├── .gitignore
│   │   ├── configs
│   │   ├── data
│   │   │   ├── rover_prepared
│   │   │   └── rover_unprepared
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── requirements
│   │   ├── scripts
│   │   │   ├── preprocessing
│   │   │   ├── optimization
│   │   │   └── uncertainty
│   │   └── ...
│   └── ...	
└── ...
```
