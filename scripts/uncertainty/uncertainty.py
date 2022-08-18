#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from ast import Pass
import os
import commentjson as json
import cv2
import numpy as np
import json
import sys
import time
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import pickle 
from common import *
from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf
from tqdm import tqdm
import pyngp as ngp # noqa

from pathlib import Path
Path("/home/ubuntu").mkdir(parents=True, exist_ok=True)


# plt.rcParams["figure.figsize"] = (40,20)



def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")
	parser.add_argument("--save_energy", default="", help="Save the energy.json file to specified path.")
	
	parser.add_argument("--calc_uncertainty", default="", help="Calculate uncertainty after reconstruction")
	parser.add_argument("--calc_energy", default="", help="Calculate energy after reconstruction") 
	parser.add_argument("--calc_PSNR", default="", help="Calculate Peak Signl to Noise ratio after single reconstruction") 
	parser.add_argument("--iterations", default=3, help="Set number of reconstructions to use for uncertainty. Default set to 3")
	parser.add_argument("--change_network", default="", help="Edit configuration file 'base.json'")


	args = parser.parse_args()
	return args

def main(x_run): #main()
	args = parse_args()

	if args.mode == "":
		if args.scene in scenes_sdf:
			args.mode = "sdf"
		elif args.scene in scenes_nerf:
			args.mode = "nerf"
		elif args.scene in scenes_image:
			args.mode = "image"
		elif args.scene in scenes_volume:
			args.mode = "volume"
		else:
			raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

#replace args.mode section below - force process to be nerf
	if args.mode == "sdf":
		mode = ngp.TestbedMode.Sdf
		configs_dir = os.path.join(ROOT_DIR, "configs", "sdf")
		scenes = scenes_sdf
	elif args.mode == "volume":
		mode = ngp.TestbedMode.Volume
		configs_dir = os.path.join(ROOT_DIR, "configs", "volume")
		scenes = scenes_volume
	elif args.mode == "nerf":
		mode = ngp.TestbedMode.Nerf
		configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
		scenes = scenes_nerf
	elif args.mode == "image":
		mode = ngp.TestbedMode.Image
		configs_dir = os.path.join(ROOT_DIR, "configs", "image")
		scenes = scenes_image

	# base_network = os.path.join(ROOT_DIR, "configs", "nerf", "base.json")
	# Need Root_Dir - sys.argv[1]

	base_network = os.path.join(configs_dir, "base.json")
	# if args.scene in scenes:
	# 	network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
	# 	base_network = os.path.join(configs_dir, network+".json")
	network = args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)
	print('Network file passed: {}'.format(network))

	
	# if args.change_network	: # try to load the given file straight away
	# 	print("Editing {}".format(network))
	# 	with open(args.screenshot_transforms) as f:
	# 		ref_transforms = json.load(f)

	
	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)

	if args.mode == "sdf":
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		scene=args.scene
		if not os.path.exists(args.scene) and args.scene in scenes:
			scene = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset"])
		testbed.load_training_data(scene)

	if args.load_snapshot:
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)

	else:
		testbed.reload_network_from_file(network)



	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)
	
		#save frame names from transforms.json 
		# every element saved as frame_0xxxx.jpg
		frames = []
		frame_count = len(ref_transforms["frames"])
		print('{} frames in transforms.json'.format(frame_count))
		for idx in range(frame_count):
			# print('Saving frame to list: {}'.format(idx))
			f = ref_transforms["frames"][int(idx)]
			frames.append(os.path.basename(f["file_path"]))
		print('Frames saved to list: {}\n'.format(frames))
	else: 
		print('No path entered for screenshot_transforms')
		sys.exit(1)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw*sh > 1920*1080*4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh)

	testbed.shall_train = args.train if args.gui else True

	testbed.nerf.render_with_camera_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]
	if args.mode == "sdf":
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Optionally match nerf paper behaviour and train on a
		# fixed white bg. We prefer training on random BG colors.
		# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
		# testbed.nerf.training.random_bg_color = False


	if (args.calc_energy or args.calc_PSNR): #args.calc_uncertainty or 
		# Timer 
		u_start_time = time.time()	

		old_training_step = 0
		n_steps = args.n_steps
		if n_steps < 0:
			n_steps = 10000

		#start reconstruction 
		if n_steps > 0:

			with tqdm(desc="Training", total=n_steps, unit="step") as t:
				while testbed.frame():
					if testbed.want_repl():
						repl(testbed)
					# What will happen when training is done?
					if testbed.training_step >= n_steps:
						if args.gui:
							testbed.shall_train = False
						else:
							break

					# Update progress bar
					if testbed.training_step < old_training_step or old_training_step == 0:
						old_training_step = 0
						t.reset()

					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
			#end reconstruction 
			
			#save screenshots of reconstruction @ actual camera positions (extracted from transforms.json)  
			testbed.fov_axis = 0
			testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi


			# extract camera "transform matrix"  from each frame 
			frame_list = [] 
			for idx in range(frame_count):
				print('Saving frame {}'.format(idx))
				print('Saving frame {}'.format(frames[idx]))

				f = ref_transforms["frames"][int(idx)]
				cam_matrix = f["transform_matrix"]
				print('x_run', x_run)
				out_dir = os.path.join(args.scene, 'mesh_' + str(x_run), 'images')
				outname = os.path.join(out_dir, os.path.basename(f["file_path"]))
				Path(out_dir).mkdir(parents=True, exist_ok = True)

				# f = ref_transforms["frames"][int(idx)]
				# cam_matrix = f["transform_matrix"]
				
				#set current reconstruction to transform matrix from frame idx
				testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
				
				#render reconstruction snapshot as numpy array with shape = frames[idx].shape 
				rendering_start_time = time.time()
				# returns array of shape (1920, 1440, 4)
				image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
				rendering_end_time = time.time()
				rendering_run_time_sec = round((rendering_end_time - rendering_start_time), 2)
				print('Time to render image: {}'.format(rendering_run_time_sec))

				saving_start_time = time.time()
				write_image(outname, image) 
				saving_end_time = time.time()
				saving_run_time_sec = round((saving_end_time - saving_start_time), 2)
				print('Time to save image: {}'.format(saving_run_time_sec))
				print('Saved as: {}'.format(outname))
				
				# save reconstruction snapshot to list
				frame_list.append(image)
				# print('Image shape: {}'.format(image.shape))
				#1920,1440,4
				

				print('Saving successful, frame {} saved\n'.format(idx))
			
				# up to this point, we have reconstruction i and all frames from transforms.json saved in a list as numpy arrays  
			print('Frames added to frame_list\nFrame_list size: {}'.format(len(frame_list)))
			frame_array = np.asarray(frame_list)
			print('List successfully converted to array\nShape: {}'.format(frame_array.shape))
			# Shape: (3, 1920, 1440, 4)

			# 1. reconstruction i is complete
			# 2. capture snapshot of reconstruction from every camera parameter in scene data (use transform.json has all frames + parameters) 
			# 3. append snapshots to frame_lists  
			# 4. convert list to numpy array, element i is frame i

	elif args.calc_uncertainty: 

		# Timer 
		u_start_time = time.time()	

		old_training_step = 0
		n_steps = args.n_steps
		if n_steps < 0:
			n_steps = 10000

		#start reconstruction 
		if n_steps > 0:

			with tqdm(desc="Training", total=n_steps, unit="step") as t:
				while testbed.frame():
					if testbed.want_repl():
						repl(testbed)
					# What will happen when training is done?
					if testbed.training_step >= n_steps:
						if args.gui:
							testbed.shall_train = False
						else:
							break

					# Update progress bar
					if testbed.training_step < old_training_step or old_training_step == 0:
						old_training_step = 0
						t.reset()

					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
			#end reconstruction 
			
			#save screenshots of reconstruction @ actual camera positions (extracted from transforms.json)  
			testbed.fov_axis = 0
			testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi


			# extract camera "transform matrix"  from each frame 
			frame_list = [] 
			for idx in range(frame_count):
				print('Saving frame {}'.format(idx))
				print('Saving frame {}'.format(frames[idx]))

				f = ref_transforms["frames"][int(idx)]
				cam_matrix = f["transform_matrix"]
				# print('x_run', x_run)
				# out_dir = os.path.join(args.scene, 'mesh_' + str(x_run), 'images')
				# outname = os.path.join(out_dir, os.path.basename(f["file_path"]))
				# Path(out_dir).mkdir(parents=True, exist_ok = True)

				# f = ref_transforms["frames"][int(idx)]
				# cam_matrix = f["transform_matrix"]
				
				#set current reconstruction to transform matrix from frame idx
				testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
				
				#render reconstruction snapshot as numpy array with shape = frames[idx].shape 
				rendering_start_time = time.time()
				# returns array of shape (1920, 1440, 4)
				image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
				rendering_end_time = time.time()
				rendering_run_time_sec = round((rendering_end_time - rendering_start_time), 2)
				print('Time to render image: {}'.format(rendering_run_time_sec))

				# saving_start_time = time.time()
				# write_image(outname, image) 
				# saving_end_time = time.time()
				# saving_run_time_sec = round((saving_end_time - saving_start_time), 2)
				# print('Time to save image: {}'.format(saving_run_time_sec))
				# print('Saved as: {}'.format(outname))
				
				# save reconstruction snapshot to list
				frame_list.append(image)
				# print('Image shape: {}'.format(image.shape))
				#1920,1440,4
				

				print('Saving successful, frame {} saved\n'.format(idx))
			
				# up to this point, we have reconstruction i and all frames from transforms.json saved in a list as numpy arrays  
			print('Frames added to frame_list\nFrame_list size: {}'.format(len(frame_list)))
			frame_array = np.asarray(frame_list)
			print('List successfully converted to array\nShape: {}'.format(frame_array.shape))
			# Shape: (3, 1920, 1440, 4)

			# 1. reconstruction i is complete
			# 2. capture snapshot of reconstruction from every camera parameter in scene data (use transform.json has all frames + parameters) 
			# 3. append snapshots to frame_lists  
			# 4. convert list to numpy array, element i is frame i




	else:

		old_training_step = 0
		n_steps = args.n_steps
		if n_steps < 0:
			n_steps = 10000

		if n_steps > 0:
			with tqdm(desc="Training", total=n_steps, unit="step") as t:
				while testbed.frame():
					if testbed.want_repl():
						repl(testbed)
					# What will happen when training is done?
					if testbed.training_step >= n_steps:
						if args.gui:
							testbed.shall_train = False
						else:
							break

					# Update progress bar
					if testbed.training_step < old_training_step or old_training_step == 0:
						old_training_step = 0
						t.reset()

					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step

			#save screenshots of reconstruction @ actual camera positions (extracted from transforms.json)  
			testbed.fov_axis = 0
			testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi

			# frame_list = [] 
			for idx in range(frame_count):
				# print('Saving frame {}'.format(idx))
				print('Saving frame {}'.format(frames[idx]))
				f = ref_transforms["frames"][int(idx)]
				cam_matrix = f["transform_matrix"]
				testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])

				outname = os.path.join(out_dir, os.path.basename(f["file_path"]))
				
				print(f"rendering {outname}")
				
				rendering_start_time = time.time()
				image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
				# write_image(outname, image) 
				rendering_end_time = time.time()
				rendering_run_time_sec = round((rendering_end_time - rendering_start_time), 2)
				
				print('Time to render image: {}'.format(rendering_run_time_sec))				
				print('Saving successful, frame {} saved\n'.format(frames[idx]))
		


	if args.save_snapshot:
		print("Saving snapshot ", args.save_snapshot)
		testbed.save_snapshot(args.save_snapshot, False)

	if args.test_transforms:
		print("Evaluating test transforms from ", args.test_transforms)
		with open(args.test_transforms) as f:
			test_transforms = json.load(f)
		data_dir=os.path.dirname(args.test_transforms)
		totmse = 0
		totpsnr = 0
		totssim = 0
		totcount = 0
		minpsnr = 1000
		maxpsnr = 0

		# Evaluate metrics on black background
		testbed.background_color = [0.0, 0.0, 0.0, 1.0]

		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.rendering_min_transmittance = 1e-4

		testbed.fov_axis = 0
		testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
		testbed.shall_train = False

		with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
			for i, frame in t:
				p = frame["file_path"]
				if "." not in p:
					p = p + ".png"
				ref_fname = os.path.join(data_dir, p)
				if not os.path.isfile(ref_fname):
					ref_fname = os.path.join(data_dir, p + ".png")
					if not os.path.isfile(ref_fname):
						ref_fname = os.path.join(data_dir, p + ".jpg")
						if not os.path.isfile(ref_fname):
							ref_fname = os.path.join(data_dir, p + ".jpeg")
							if not os.path.isfile(ref_fname):
								ref_fname = os.path.join(data_dir, p + ".exr")

				ref_image = read_image(ref_fname)

				# NeRF blends with background colors in sRGB space, rather than first
				# transforming to linear space, blending there, and then converting back.
				# (See e.g. the PNG spec for more information on how the `alpha` channel
				# is always a linear quantity.)
				# The following lines of code reproduce NeRF's behavior (if enabled in
				# testbed) in order to make the numbers comparable.
				if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
					# Since sRGB conversion is non-linear, alpha must be factored out of it
					ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
					ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
					ref_image[...,:3] *= ref_image[...,3:4]
					ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
					ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])

				if i == 0:
					write_image("ref.png", ref_image)

				testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
				image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

				if i == 0:
					write_image("out.png", image)

				diffimg = np.absolute(image - ref_image)
				diffimg[...,3:4] = 1.0
				if i == 0:
					write_image("diff.png", diffimg)

				A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
				R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
				mse = float(compute_error("MSE", A, R))
				ssim = float(compute_error("SSIM", A, R))
				totssim += ssim
				totmse += mse
				psnr = mse2psnr(mse)
				totpsnr += psnr
				minpsnr = psnr if psnr<minpsnr else minpsnr
				maxpsnr = psnr if psnr>maxpsnr else maxpsnr
				totcount = totcount+1
				t.set_postfix(psnr = totpsnr/(totcount or 1))

		psnr_avgmse = mse2psnr(totmse/(totcount or 1))
		psnr = totpsnr/(totcount or 1)
		ssim = totssim/(totcount or 1)
		print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")

	if args.save_mesh:
		mesh_outpath = '/'.join(args.save_mesh.split("/")[:-1])
		Path(mesh_outpath).mkdir(parents=True, exist_ok = True)
		res = args.marching_cubes_res or 256
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}]")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res])

	return frame_array


def main_loop(n):
	'''
	Creates an iterable process for the reconstructions 
		Parameters: 
			n (int): Number of reconstruction to create; defaults to three

		Returns:
			array of all reconstructions with all respective input frames
			shape: (B,P,W,H,C), where B is number of reconstructions, P is number of images in dataset
	''' 
	cmds = parse_args()

	# if (cmds.calc_energy or cmds.calc_PSNR): 
		# recon = []
	for i in range(1,n+1):
		main(i)
		print('Reconstruction {} complete\n'.format(i))
			# recon.append(x) 
		# recon_array = np.asarray(recon)
		# print('Shape of reconstrucion array: {}'.format(recon_array.shape))
		# Shape of reconstrucion array: (2, 3, 1920, 1440, 4)

		# return recon_array

	# elif cmds.calc_uncertainty:
	# 	recons = []
	# 	for i in range(1,n+1):
	# 		#x_i is reconstruction i; array of len(frames from transforms.json) 
	# 		#x_i = tensor of size(P,W,H,C), where P = number of images in dataset
	# 		x = main()
	# 		print('Reconstruction {} complete\n'.format(i))
	# 		recons.append(x) 
	# 	recons_array = np.asarray(recons)
	# 	print('Shape of reconstrucion array: {}'.format(recons_array.shape))
	# 	# Shape of reconstrucion array: (2, 3, 1920, 1440, 4)

	
	# else: 
	# 	print('Error with main_loop(n)')
	# 	pass
# 
	# return recons_array


def read_actual_images():
	''' 
		Go to input data scene, extract each frame path from transform.json file, read into memory using cv2 library 
			
			Parameters: 
				N/A
			Return: 
				Ground_truth_array: array with every actual scan as grayscale np.array
				Frame_ID: List of frame names
	'''

	cmds = parse_args()

	print('Ground truth images extracted from: {}\n'.format(cmds.scene))
	# Ground truth images extracted from: /home/ubuntu/data/rover_2_subset_2

	ref_transforms = {}
	if cmds.screenshot_transforms:
		with open(cmds.screenshot_transforms) as f:
			ref_transforms = json.load(f)
	
	frame_count = len(ref_transforms["frames"])
	 
	# save frame names from transforms.json into frame_ID
		# syntax: [frame_0xxxx.jpg, frame_0xxxy.jpg, ...]
	# read actual input frames from transforms.json file using cv2.imread() 
		#save into ground_truth_images list as an array 
	# compute pixel-wise absolute error and square error between input frames and reconstruction screenshot
	# standardize loss to pixel-wise loss
		# width * height * color_channel * scan_count
	# total energy of reconstruction (float) = sum of pixel-wise loss
	
	print('Reading ground truth images into memory')
	
	# convert images to grayscale if taking PSNR 
	if cmds.calc_PSNR:
		frame_ID = []
		ground_truth_images = []
		# frame_count = len(ref_transforms["frames"])

		for idx in range(frame_count):
			f = ref_transforms["frames"][int(idx)]
			frame_ID.append(os.path.basename(f["file_path"]))
			
			actual_scan = cv2.imread(cmds.scene+f["file_path"][1:])
			grayscale_actual_scan = np.mean(actual_scan, axis = -1)
			ground_truth_images.append(grayscale_actual_scan)
	
	# read in images as RBG
	else:
		frame_ID = []
		ground_truth_images = []
		# frame_count = len(ref_transforms["frames"])
		for idx in range(frame_count):
			f = ref_transforms["frames"][int(idx)]
			frame_ID.append(os.path.basename(f["file_path"]))		

			actual_scan = cv2.imread(cmds.scene+f["file_path"][1:])
			ground_truth_images.append(actual_scan)

	ground_truth_array = np.asarray(ground_truth_images)
	print('Ground Truth array shape: {}'.format(ground_truth_array.shape))

	return ground_truth_array, frame_ID #frame_count,

	
def read_recon_images(i): #/
	''' 
		Go to saved reconstruction data scene and read into memory using cv2 library 
			
			Parameters: 
				N/A
			Return: 
				frame_array: array with every reconstruction 
				Frame_ID: List of frame names
	'''

	cmds = parse_args()


	out_dir = os.path.join(cmds.scene, 'mesh_' + str(i))
	# outname = os.path.join(out_dir, os.path.basename(f["file_path"]))
	# Path(out_dir).mkdir(parents=True, exist_ok = True)

	print('Reconstruction images extracted from: {}\n'.format(out_dir))
	# Ground truth images extracted from: /home/ubuntu/data/rover_2_subset_2

	ref_transforms = {}
	if cmds.screenshot_transforms:
		with open(cmds.screenshot_transforms) as f:
			ref_transforms = json.load(f)
	
	frame_count = len(ref_transforms["frames"])
	 
	# save frame names from transforms.json into frame_ID
		# syntax: [frame_0xxxx.jpg, frame_0xxxy.jpg, ...]
	# read actual input frames from transforms.json file using cv2.imread() 
		#save into ground_truth_images list as an array 
	# compute pixel-wise absolute error and square error between input frames and reconstruction screenshot
	# standardize loss to pixel-wise loss
		# width * height * color_channel * scan_count
	# total energy of reconstruction (float) = sum of pixel-wise loss
	
	print('Reading reconstruction images into memory')
	
	# convert images to grayscale if taking PSNR 
	if cmds.calc_PSNR:
		pass
		# frame_ID = []
		# recon_images = []
		# # frame_count = len(ref_transforms["frames"])

		# for idx in range(frame_count):
		# 	f = ref_transforms["frames"][int(idx)]
		# 	frame_ID.append(os.path.basename(f["file_path"]))
			
		# 	actual_scan = cv2.imread(cmds.scene+f["file_path"][1:])
		# 	grayscale_actual_scan = np.mean(actual_scan, axis = -1)
		# 	recon_images.append(grayscale_actual_scan)
	
	# read in images as RBG
	elif cmds.calc_energy:
		
		frame_ID = []
		recon_images = []
		# frame_count = len(ref_transforms["frames"])
		for idx in range(frame_count):
			f = ref_transforms["frames"][int(idx)]
			# print('f["file_path"]:{}'.format(f["file_path"][1:]))
			frame_ID.append(os.path.basename(f["file_path"][1:]))		

			scan = cv2.imread(out_dir+f["file_path"][1:])
			recon_images.append(scan)

		recon_images_array = np.asarray(recon_images)
		print('Reconstruction array shape: {}'.format(recon_images_array.shape))
	
	elif cmds.calc_uncertainty:
		frame_array = []




	return recon_images_array, frame_ID #frame_count,


def calc_uncertainty(frame_array, frame_ID): # (frame_array,frame_ID): # (frame_array): (frame_ID):
	'''
		Calculate pixel-wise standard deviation of all identical frames between all reconstructions

			Parameters:
				frame_array: array with all reconstructions and all respetive input frames as np.arrays
				ground_truth_images: list of all actual scans as np.arrays
				frame_count: number of actuals scans in transforms.json
			
			Returns:
				dictionary with frame names as keys, standard deviation (np.array) as values 

	'''
	# 

	print('\nCalculating uncertainty\nInput shape: {}'.format(frame_array.shape))	
	# Input shape: (2, 3, 1920, 1440, 4)
	cmds = parse_args()

	out_dir = os.path.join(cmds.scene,'uncertainty')
	Path(out_dir).mkdir(parents=True, exist_ok = True)
	
	# ref_transforms = {}
	# if cmds.screenshot_transforms:
	# 	with open(cmds.screenshot_transforms) as f:
	# 		ref_transforms = json.load(f)
	 
	# 	#save frame names from transforms.json 
	# 	# every element saved as frame_0xxxx.jpg
	# 	frame_ID = []
	# 	frame_count = len(ref_transforms["frames"])
	# 	for idx in range(frame_count):
	# 		f = ref_transforms["frames"][int(idx)]
	# 		frame_ID.append(os.path.basename(f["file_path"]))
	

	frame1 = {}

	#loop through every frame
	for i in range(frame_array.shape[1]):
		#loop through each reconstruction
		temp_frames = [] 
		for j in range(frame_array.shape[0]):
			temp_frames.append(frame_array[j][i])
		
		y = np.asarray(temp_frames)
		# print('Shape of temp_frames array: {}'.format(y.shape))
			# (2, 1920, 1440, 4)
			# temp_frames created 3 times (3 scans) 
			# each stacked array has 2 frames (2 reconstructions)
		
		z = np.std(y, axis = 0)
		
		frame1[frame_ID[i]] = z

	# print('Shape of z array: {}'.format(z.shape))
	# (1920, 1440, 3)
	return frame1




def calc_energy(frame_array, ground_truth_images): #  #(frame_array):
	'''
		Calculate pixel-wise Least Absolute Deviation, Lease Square Error of entire reconstruction 

			Parameters:
				frame_array: array with all reconstructions and all respetive input frames as np.arrays
				ground_truth_images: list of all actual scans as np.arrays
				frame_count: number of actuals scans in transforms.json
			
			Returns:
				Total energy (float) per reconstruction  

	'''
	
	cmds = parse_args()

	frame_count = ground_truth_images.shape[0]
	print('\nCalculating energy\nInput shape: {}'.format(frame_array.shape))
	# If run with uncertainty: Input shape: (3, 3, 1920, 1440, 4)
	# If only run with energy: Input shape: (3, 1920, 1440, 3)

	# base_images = '/'.join(cmds.screenshot_transforms.split("/")[:-1])
	# print('Ground truth images extracted from: {}\n'.format(base_images))
	# # Ground truth images extracted from: /home/ubuntu/data/rover_2_subset_2
	

	out_dir = os.path.join(cmds.scene,'energy')
	Path(out_dir).mkdir(parents=True, exist_ok = True)
	
	# ref_transforms = {}
	# if cmds.screenshot_transforms:
	# 	with open(cmds.screenshot_transforms) as f:
	# 		ref_transforms = json.load(f)
	 
	# save frame names from transforms.json into frame_ID
		# syntax: [frame_0xxxx.jpg, frame_0xxxy.jpg, ...]
	# read actual input frames from transforms.json file using cv2.imread() 
		#save into ground_truth_images list as an array 
	# compute pixel-wise absolute error and square error between input frames and reconstruction screenshot
	# standardize loss to pixel-wise loss
		# width * height * color_channel * scan_count
	# total energy of reconstruction (float) = sum of pixel-wise loss

	# frame_ID = []
	# ground_truth_images = []
	# frame_count = len(ref_transforms["frames"])
	# for idx in range(frame_count):
	# 	f = ref_transforms["frames"][int(idx)]
	# 	frame_ID.append(os.path.basename(f["file_path"]))
	# 	# print('\nFrame ID added: {}\nReading ground truth image into memory'.format(frame_ID[idx]))
		
	# 	start_scan_time = time.time()
	# 	actual_scan = cv2.imread(base_images+f["file_path"][1:])
	# 	end_scan_time = time.time()
	# 	ground_truth_images.append(actual_scan)
	# 	# print('Ground truth loaded in {} seconds'.format(round(end_scan_time - start_scan_time),2))
	# 	# 0 seconds 
		

	total_energy1 = 0.0
	total_energy2 = 0.0

	#modify - remove additional reconstruction, use first one from calc_uncertainty call of main_loop()
	# if (cmds.calc_uncertainty and cmds.calc_energy):
	# print('Entering if statement: cmds.calc_uncertainty and cmds.calc_energy')
	# calc_energy will take first_reconstruction from main_loop;  recons[0]
	# no need to run additional reconstruction
	# this will reduce the 5D array from main_loop to 4D

	# Standardize Energy 
	# Energy will be calculate for data sets of varying size
	# standardize loss by dividing total energy by total number of pixels from input scans in full scene 
	# count of pixels in one full scene = width, height, color channel, number of scans
	scans = frame_array.shape[0]
	pixel_width = frame_array.shape[1]
	pixel_height = frame_array.shape[2]
	color_channel = frame_array.shape[3] #-1 # exclude alpha channel
	scale_factor = scans * pixel_width * pixel_height * color_channel
	# (1, 3, 1920, 1440, 4)
	# print('Scale factor: {}'.format(scale_factor))
	# Scale factor: 24883200

	for i in range(frame_count):
		x = frame_array[i]#[...,:3]
		# print('\nShape of frame i from reconstruction: {}'.format(x.shape))
		# (1920, 1440, 3)
		y = ground_truth_images[i]
		# print('Shape of frame i from ground truth: {}'.format(y.shape))
		# (1920, 1440, 3)

		# Element wise losses
		# L1_e = np.abs(x - y)
		# L2_e = np.abs(x - y)**2
		# # L2_e = np.sum((x - y)**2)
		# print('Element-wise\t L1_e shape: {}\t L2_e shape: {}'.format(L1_e.shape, L2_e.shape))
		# Element-wise     L1_e shape: (1920, 1440, 3)     L2_e shape: (1920, 1440, 3)


		# redo but normalize losses
		
		x_flatten = x.flatten()
		y_flatten = y.flatten()
		# print('Flatten x shape: {}\n Flatten y shape: {}'.format(x_flatten.shape, y_flatten.shape))
		# Flatten x shape: (8294400,)
		# Flatten y shape: (8294400,)


		L1_e2 = np.abs(x_flatten - y_flatten) 
		L2_e2 = np.abs(x_flatten - y_flatten)**2

		# print('New L1 and L2 flattened shape: {}, {}'.format(L1_e2.shape, L2_e2.shape))
		# New L1 and L2 flattened shape: (8294400,), (8294400,)

		L1_norm = L1_e2 / float(np.max(L1_e2)) 
		L2_norm = L2_e2 / float(np.max(L2_e2))
		# print(x, L1_e2)
		# print('L1_norm: {}\nL2_norm: {}'.format(L1_norm, L2_norm))
		L1_norm2 = (np.max(L1_e2)) 
		L2_norm2 = (np.max(L2_e2))
		# print('L1_norm2: {}\nL2_norm2: {}'.format(L1_norm2, L2_norm2))


		# print('L1_norm shape: {}\nL2_norm shape: {}'.format(L1_norm.shape, L2_norm.shape))
		# L1_norm shape: (8294400,)
		# L2_norm shape: (8294400,)
		
		L1_avg = np.sum(L1_norm) / float(L1_norm.shape[0])
		L2_avg = np.sum(L2_norm) / float((L1_norm.shape[0]))

		# print('L1_avg: {}\nL2_avg: {}'.format(L1_avg, L2_avg))

		# print('L1_avg shape: {}\nL2_avg shape: {}'.format(L1_avg.shape, L2_avg.shape))

		# #image-wise losses
		# L1_i = np.abs(x - y).sum() 
		# 	# divide by number pixels in image to get avg L1_i error per image
		# 	# divide by max pixel to normlalize 
		# L2_i = np.abs((x - y)**2).sum() # 

		# energy = round((L1_i+L2_i)/(scale_factor),3)
		energy2 = round((L1_avg+L2_avg)/(2),3)

		# print('Previous Energy: {}\nNew Energy: {}'.format(energy,energy2))
		# print('Energy: {}'.format(energy2))

		# total_energy1 += energy
		total_energy2 += energy2
	
	# total_energy1 = total_energy1/frame_count
	total_energy2 = total_energy2/frame_count

	# print('Total Energy 1: {}\nTotal Energy 2: {}\n\n'.format(total_energy1, total_energy2))
	print('Total Energy: {}\n'.format(total_energy2))

###############Start##########################
	# #leave as is
	# elif cmds.calc_energy:
	# 	print('Entering if statement: cmds.calc_energy')
	# 	# Standardize Energy 
	# 	# Energy will be calculate for data sets of varying size
	# 	# standardize loss by dividing total energy by total number of pixels from input scans in full scene 
	# 	# count of pixels in one full scene = width, height, color channel, number of scans
	# 	scans = frame_array.shape[1]
	# 	pixel_width = frame_array.shape[2]
	# 	pixel_height = frame_array.shape[3]
	# 	color_channel = frame_array.shape[4] -1 # exclude alpha channel
	# 	scale_factor = scans * pixel_width * pixel_height * color_channel

	# 	# frame_array is input to calc_energy 
	# 	# array of reconstruction snapshots  
	# 	for i in range(frame_count):
	# 		x = frame_array[0][i][...,:3]
	# 		# print('\nShape of frame i from reconstruction: {}'.format(x.shape))
	# 		# (1920, 1440, 3)
	# 		y = ground_truth_images[i]
	# 		# print('Shape of frame i from ground truth: {}'.format(y.shape))
	# 		# (1920, 1440, 3)


	# 		# # Element wise losses
	# 		# L1_e = np.abs(x - y)
	# 		# L2_e = np.abs(x - y)**2
	# 		# # L2_e = np.sum((x - y)**2)
	# 		# print('Element-wise\t L1_e shape: {}\t L2_e shape: {}'.format(L1_e.shape, L2_e.shape))
	# 		# Element-wise     L1_e shape: (1920, 1440, 3)     L2_e shape: (1920, 1440, 3)


	# 		# redo but normalize losses
			
	# 		x_flatten = x.flatten()
	# 		y_flatten = y.flatten()
	# 		# print('Flatten x shape: {}\n Flatten y shape: {}'.format(x_flatten.shape, y_flatten.shape))
	# 		# Flatten x shape: (8294400,)
	# 		# Flatten y shape: (8294400,)


	# 		L1_e2 = np.abs(x_flatten - y_flatten) 
	# 		L2_e2 = np.abs(x_flatten - y_flatten)**2

	# 		print('New L1 and L2 flattened shape: {}, {}'.format(L1_e2.shape, L2_e2.shape))
	# 		# New L1 and L2 flattened shape: (8294400,), (8294400,)

	# 		L1_norm = L1_e2 / float(np.max(L1_e2)) 
	# 		L2_norm = L2_e2 / float(np.max(L2_e2))

	# 		print('L1_norm shape: {}\nL2_norm shape: {}'.format(L1_norm.shape, L2_norm.shape))
	# 		# L1_norm shape: (8294400,)
	# 		# L2_norm shape: (8294400,)
			
	# 		L1_avg = np.sum(L1_norm) / float(L1_norm.shape[0])
	# 		L2_avg = np.sum(L2_norm) / float((L1_norm.shape[0]))

	# 		print('L1_avg: {}\nL2_avg: {}'.format(L1_avg, L2_avg))

	# 		# print('L1_avg shape: {}\nL2_avg shape: {}'.format(L1_avg.shape, L2_avg.shape))

	# 		# #image-wise losses
	# 		# L1_i = np.abs(x - y).sum() 
	# 		# 	# divide by number pixels in image to get avg L1_i error per image
	# 		# 	# divide by max pixel to normlalize 
	# 		# L2_i = np.abs((x - y)**2).sum() # 

	# 		# energy = round((L1_i+L2_i)/(scale_factor),3)
	# 		energy2 = round((L1_avg+L2_avg)/(2),3)

	# 		# print('Previous Energy: {}\nNew Energy: {}'.format(energy,energy2))
	# 		print('Energy: {}'.format(energy2))

	# 		# total_energy1 += energy
	# 		total_energy2 += energy2
		
	# 	# total_energy1 = total_energy1/frame_count
	# 	total_energy2 = total_energy2/frame_count
	
	# 	# print('Total Energy 1: {}\nTotal Energy 2: {}'.format(total_energy1, total_energy2))
############### end ##########################

	# else:
	# 	print('Error with calc_energy function')
	# 	pass

	print('Total Energy: {}'.format(total_energy2))

	return out_dir, total_energy2 ,frame_count


# input is an array with all frames from one reconstruction
# Parameters: 
# 	recon_array: array output from main_loop(), contains all snapshots of current reconstruction 
	# ground_truth_array: output from read_images, contains all actual scans as grayscale array
	# frame_count: number of frames in data # replace with len(frame_ID)
	# frame_ID: name of frames in data 

# input = output of main()
def calc_PSNR(recon_array, ground_truth_array, frame_count, frame_ID): # (x): 

	cmds = parse_args()
	print('\nSaving PSNR')
	
	out_dir = os.path.join(cmds.scene,'PSNR')
	Path(out_dir).mkdir(parents=True, exist_ok = True)


	# Converting reconstruction images into grayscale, save as array 
	# calc_PSNR will always execute with only one reconstruction -  recon_array.shape[0] should always equals 1
	reconstruction_images = []

	for r in range(recon_array.shape[0]):
		for i in range(frame_count):
			y = recon_array[r][i][...,:3]
			z = np.mean(y, axis = -1)
			# print('Shape of grayscale image: {}'.format(z.shape))
			# Every z is a new grayscale image from reconstruction
			reconstruction_images.append(z)

	gs_recon_array = np.asarray(reconstruction_images)

	# Scale factor to be used in MSE 
	scale_factor = ground_truth_array.shape[0] * ground_truth_array.shape[1] * ground_truth_array.shape[2]
	print('Scale Factor: {}\n'.format(scale_factor))

	# ground_truth_images = ground_truth_array.tolist()

	psnr_dict = {}
	PSNR = 0

	# compare actual scans to reconstruction snapshots
	# calculate MSE, MAX pixel value for every actual image
	 
	for i in range(frame_count):
		x = ground_truth_array[i] 
		y = gs_recon_array[i]
		# i is for ground truth, j is for reconstruction
		# print(i, j)
		# print('Len of i: {}\n Len of j: {}'.format(len(i), len(j)))
		#1920, 1920
		print('Shape of x: {}\nShape of y: {}\n'.format(x.shape, y.shape))
		# AttributeError: 'list' object has no attribute 'shape'

		# Image-wise L1 loss
		L1_image = np.abs(x - y).sum()
		print("L1 calculated for PSNR: ", L1_image)

		# MSE calculated per image
		# MSE_i = ((L1_image)**2)/scale_factor
		MSE_i = ((np.abs(x - y))**2).sum()/scale_factor		
		print('MSE_i: {}'.format(MSE_i))

		# Find max values from ground_truth_images 
		flat_array = x.flatten()
		MAX_i = np.max(flat_array)
		print('Max pixel value of ground truth image: {}\nMax squared: {}'.format(MAX_i,MAX_i**2 ))

		psnr_image = 10 * np.log10((MAX_i**2)/MSE_i)
		print('PSNR image value output: {}\n'.format(psnr_image))
		
		psnr_dict[frame_ID[i]] = psnr_image
		PSNR += psnr_image

	print('PSNR successful, total PSNR: {}'.format(PSNR))

	return out_dir, PSNR


def save_uncertainty(frame_dict): #(dictionary, path)
	'''
		Save standard deviation for both texture and geometry as images using pyplot library
			
			Parameters: 
				frame_dict: input dictionary from calc_uncertainty
					dictionary has frame names as keys, std deviation as value for each frame
		 
		 	Return:

	'''
	
	cmds = parse_args()
	
	frame_ID = [x for x in frame_dict.keys()]

	# Texture Uncertainty 
	out_dir_texture = os.path.join(cmds.scene,'uncertainty','images','texture')
	Path(out_dir_texture).mkdir(parents=True, exist_ok = True)
	
	for i in range(len(frame_ID)):
		frame = str(frame_ID[i])
		out_name = os.path.join(out_dir_texture, frame)
		y = frame_dict[frame_ID[i]][...,:3]
		# print('Frame: {}'.format(frame))
		plt.imsave(out_name, y, dpi = 90)

	# Geometric Uncertainty 
	out_dir_geo = os.path.join(cmds.scene,'uncertainty','images','geometry')
	Path(out_dir_geo).mkdir(parents=True, exist_ok = True)

	
	for i in range(len(frame_ID)):
		frame = str(frame_ID[i])
		out_name = os.path.join(out_dir_geo , frame)
		y = frame_dict[frame_ID[i]][...,:3]
		# print('Frame: {}\nShape: {}'.format(frame, y.shape))
		# Frame: frame_00003.jpg
		# Shape: (1920, 1440, 3)

		z = np.mean(y, axis = -1)
		# print('Shape of array mean: {}'.format(z.shape))
		# Shape of array mean: (1920, 1440)

		plt.imsave(out_name, z, dpi = 90) 


	base_dir ='/'.join(out_dir_texture.split("/")[:-1])
	print('Uncertainty images saved in: {}'.format(base_dir))
	# Uncertainty images saved in: /home/ubuntu/data/rover_2_subset_2/uncertainty/images
	


def save_energy(a, b, c, d):
	''' Save energy to json'''

	e_end_time = time.time()
	e_run_time_sec = round((e_end_time - c), 2)
	e_run_time_min = round((e_end_time - c)/60, 2)

	data = {
		"scene" : a,
		"frames" : d, 
		"energy" : b, 
		"time(sec)" : e_run_time_sec,
		"time(min)" : e_run_time_min
	}

	json_data = json.dumps(data, indent = 4) 
	file = 'energy.json'
	json_file = os.path.join(a, file)
	with open(json_file, 'w') as out:
			out.write(json_data)
			print('\nEnergy data successfully written to: {}'.format(json_file))



def save_PSNR(a, b, c):
	''' Save PSNR for given scene to json'''

	cmds = parse_args()
	network = cmds.network
	network2 = network.split('/')[-1]
	# "/home/ubuntu/repos/mars-metaverse/configs/base.json"
	# 
	steps = cmds.n_steps	
	near_distance = cmds.near_distance



	data = {
		"Network Setting used" : network2,
		"Number_Scans" : b,
		"Training steps" : steps,
		"Distance Arg" : near_distance,
		"PSNR" : c
	}

	json_data = json.dumps(data, indent = 4) 
	# file = 'PSNR.json'
	file = str('PSNR_' + network2)
	json_file = os.path.join(a, file)
	with open(json_file, 'w') as out:
			out.write(json_data)
			print('PSNR data successfully written to: {}'.format(json_file))


#plot standard deviation as images during run time 
def plot_output(x):
	frame_ID = [x for x in x.keys()]
	for i in range(len(frame_ID)):
		y = x[frame_ID[i]][...,0:3]
		plt.title(frame_ID[i]);plt.imshow(y);plt.show()



def uncertainty_energy_main():
	start_time = time.time()
	print('\nRunning training with uncertainty and energy')
	
	recons = main_loop(int(cmds.iterations))
	print('Reconstructions complete')
	frames_dict = calc_uncertainty(recons)
	print('Uncertainty complete')	
	save_uncertainty(frames_dict)
	print('Uncertainty complete, Calculating energy\n')
	
	# only need one reconstruction to calculate energy 
	# avoid additional reconstruction, use first one from main_loop call above
	first_recons = recons[0]


	path, energy = calc_energy(first_recons)
	print('Energy complete')
	save_energy(path, energy, start_time)


def uncertainty_main():
	print('Running training with uncertainty')
	recons = main_loop(int(cmds.iterations))
	print('Reconstructions complete')
	ground_truth_array, frame_ID = read_actual_images()
	frames_dict = calc_uncertainty(recons,frame_ID)
	print('Uncertainty complete')	
	save_uncertainty(frames_dict)


def energy_main():
	start_time = time.time()
	print('Running training with energy')
	#calculating energy of reconstruction - only need 1 reconstruction 
	# recons = main_loop(1)

	main(1)
	print('Reconstruction complete')
	# ground_truth_array, frame_ID = read_images()
	ground_truth_array, frame_ID = read_actual_images()
	recon_images_array, frame_ID = read_recon_images(1)
	
	path, energy, frame_count = calc_energy(recon_images_array, ground_truth_array)
	print('Energy complete')
	save_energy(path, energy, start_time, frame_count)


def PSNR_Main():
	print('Running training with PSNR')
	recons = main_loop(1)
	ground_truth_array, frame_count, frame_ID = read_images()
	out_dir, PSNR = calc_PSNR(recons, ground_truth_array, frame_count, frame_ID)
	save_PSNR(out_dir, frame_count, PSNR)



if __name__ == "__main__":
	start_time = time.time()
	cmds = parse_args()

	if (cmds.calc_uncertainty and cmds.calc_energy):
		uncertainty_energy_main()			

	elif cmds.calc_uncertainty:
		uncertainty_main()
		
	elif cmds.calc_energy:
		energy_main()

	elif cmds.calc_PSNR:
		PSNR_Main()

	else:
		print('Running training without uncertainty or energy')
		main()                              
	
	end_time = time.time()
	run_time_sec = round((end_time - start_time), 2)
	run_time_min = round((end_time - start_time)/60, 2)
	try:
		print('Script executes in {} seconds, {} minutes'.format(run_time_sec, run_time_min))
	except:
		print('Error with time function')

