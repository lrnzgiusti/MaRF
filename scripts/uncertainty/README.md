## Uncertainty


The uncertainty of the model introduces an additional performance metric to our process. It allows users to identify viewpoints within the reconstruction 
where the model was less confident, or more _uncertain_. Holding all parameters constant and reconstructing the same dataset multiple times should ideally generate identical results. This is not the case due to randomness in the training and as a results, there are slight variations between reconstructions. 

To calculate the uncertainty, the same reconstruction is rendered multiple times while holding all parameters constant. The pixel-wise standard deviation between all reconstructions is calculated and visualized as images. The option between grayscale and RGB standard deviations can be specified to measure either the geometric or  appearance uncertainties.

<!-- ![frame_0000](https://user-images.githubusercontent.com/32660307/170205326-b647f0f1-0890-4a5d-9e7b-577affa25b39.jpg) -->

**Inputs:** 
  - scene: Path to the '/images' folder, produced by the process_sols.py script 
  - screenshot_transforms: Path to transforms.json file, produced by the process_sols.py script 
  - iterations: *x* number of reconstructions 
  - mode: Nerf 
  - n_Steps: Number of training steps 
  - calc_uncertainty: Boolean flag to toggle uncertainty calculations on/off

**Outputs:**
  - A new directory '/uncertainty' is created under the input 'scene' path provided above 
    - The standard deviations are saved as images here
  - Normalized standard deviation of the entire scene  

**Steps to calculate the uncertainty:**
1. Run reconstruction *x* times, provided by user  
2. Using all input camera parameters from transforms.json file (output of process_sols.py), save screenshots of each reconstruction 
3. Algorithms will extract and pair matching frames from each reconstruction and calculate the pixel-wise standard deviation  
4. Once the standard deviation of each camera paramters is calculated, a fly-by of the uncertainties is rendered 
5. Lastly, a normalized uncertainty value will be calculated for the entire reconstruction 

***Note: The script requires the --screenshot_transforms argument to be passed with a path to the transforms.json file. Without this, the process can not determine what camera parameters to screenshot. Below is an example query to run the uncertainty of the rover data.***

```bash
python scripts/uncertainty.py --scene ~/data/rover_2 --mode nerf --n_steps=5000 --screenshot_transforms ~/data/rover_2_subset_2/transforms.json --calc_uncertainty True --iterations 10
```
