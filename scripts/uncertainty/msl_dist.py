import os
import re
from pyquaternion import Quaternion


radix = "https://planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_mastcamz_ops_raw/browse/sol/" #for the sake of readability
scan_1_path = radix+"00001/ids/edr/zcam/ZL0_0001_0667035659_000EBY_N0010052AUT_04096_0260LMJ03.png"
scan_2_path = radix+"00163/ids/edr/zcam/ZL0_0163_0681407452_428EDR_N0060000ZCAM08171_1100LMJ03.png"



config_scan_1_path = scan_1_path.replace("browse", "data").replace("png", "xml")
config_scan_2_path = scan_2_path.replace("browse", "data").replace("png", "xml")

filename_1 = (config_scan_1_path.split('/')[-1])
filename_2 = (config_scan_2_path.split('/')[-1])

if filename_1 not in os.listdir() or filename_2 not in os.listdir():
	os.system("wget "+ config_scan_2_path)
	os.system("wget "+ config_scan_1_path)


def get_rover_position(filename):
	camera_config = open(filename, "r").readlines()
	idx =  [i for i, item in enumerate(camera_config) if re.search('<geom:Quaternion_Plus_Direction>', item)][0]
	return tuple(re.findall(r"0\.\d+", ','.join(camera_config[idx:idx+5])))


q1_rover = Quaternion(get_rover_position(config_scan_1_path.split('/')[-1]))
q2_rover = Quaternion(get_rover_position(config_scan_2_path.split('/')[-1]))
print(Quaternion.absolute_distance(q1_rover,q2_rover))
