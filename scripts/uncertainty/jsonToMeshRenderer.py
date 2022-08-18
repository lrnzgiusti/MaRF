# -*- coding: utf-8 -*-
"""Extract parameters from JSON file 'transform.json
    For every frame value, script will return each frame with respective R and T matrix
        R: Rotation matrix of shape (N, 3, 3)
        T: Translation matrix of shape (N, 3)
            R & T:  Full transformation from World Coordinates (WC) to Normalized Device Coordinates (NDC)
    
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/cameras.html

"""

import numpy as np 
import json
import sys
import os


#function to check os.path for transform.json file
#input_file = output of function above 

cwd = os.getcwd()
select_dir = ['data']
working_dir = [i for i in os.listdir() if any(j in i for j in select_dir)][0] 
input_file = 'transforms.json'
input_path = os.path.join(working_dir, input_file) 
 

try:
    f = open(input_path)
except NameError:
    print("\n\nFile {} does not exist. Import file into working directory and try again. \n".format(input_path))
# else:
#     print('No errors, file opened')


data = json.load(f)
img = data['frames']


# print('Full JSON view:\n')
# data
# print('Looking into the frames: \n')
# img


print('\nTaking the first frame and extracting parameters: \n')
print('img[0]: \n{}\n'.format(img[0]))
print('Extracted file name: {}'.format(list(img[0].values())[0].split('/')[2].split('.')[0]))

sharpness = list(img[0].values())[1]
print('Sharpness: {}'.format(sharpness))

transform_matrix = list(img[0].values())[2]
print('Transform Matrix is composed of R ([3,3]) + T ([3,1]): \n')
print('\tR: Rotation matrix of shape (N, 3, 3) \n\tT: Translation matrix of shape (N, 3)\n')
print('Shape: {}\n'.format(np.shape(transform_matrix)))
transform_matrix

print('Logic will iterate through transform matrix of each frame,\n extract first 3 elements of each array and store as R,\n extract last element of each array and store as T\n\n')


d = {}
for i in range(len(data['frames'])):
  # f = []
  r = []
  t = []
  file_num = list(img[i].values())[0].split('/')[2].split('.')[0]

  for j in range(len(list(img[i].values())[2])-1):
    r.append(list(img[i].values())[2][j][0:3])

  for k in range(len(list(img[i].values())[2])-1):
    t.append(list(img[i].values())[2][k][3])  
  
  d[file_num] = {
      'r': r,
      't': t
  }

file_keys = list(d.keys())
n_frames = len(d)

r_values = []
t_values = []
for i in range(len(d)):
  # print(d[file_keys[i]]['r'])
  r_values.append(d[file_keys[i]]['r'])
  # print(d[file_keys[i]]['t'])
  t_values.append(d[file_keys[i]]['t'])


for i in range(len(d)):
  print('File: {}'.format(file_keys[i]) + '\n\tr: {}'.format(r_values[i]) + '\n\tt: {}'.format(t_values[i])  )




