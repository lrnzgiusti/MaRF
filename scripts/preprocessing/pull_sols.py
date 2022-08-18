import os
import sys
import wget
import requests
from bs4 import BeautifulSoup


class Sol(object):
    def __init__(self, idx, server):
        self.idx = idx
        self.label = 'SOL' + str(self.idx).zfill(5)
        self.server = server
        self.url = None
        self.sol_dir = None
        self.scenes = []
        self.params = []

        # constructing URLs
        if self.server == 'helicam':
            self.url = 'https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_helicam/browse/sol/'
            self.sol_dir = self.url + str(self.idx).zfill(5) + '/ids/edr/heli/'
        elif self.server == 'mastcam':
            self.url = 'https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz_ops_raw/browse/sol/'
            self.sol_dir = self.url + str(self.idx).zfill(5) + '/ids/edr/zcam/'

    def parser(self):
        """This function finds links of images and labels on PDS"""
        try:
            temp_req = requests.get(self.sol_dir)
            temp_soup = BeautifulSoup(temp_req.text, 'html.parser')
        except Exception as e:
            print(e)
            sys.exit(1)

        # filter the scrape to all links in SOL directory
        sol_pulls = []
        for link in temp_soup.find_all('a'):
            sol_pulls.append(link.get('href'))

        # filtering links to IMG and LBL files
        sol_pics = []
        sol_labels = []
        try:
            for x in sol_pulls:
                if ('IMG' in x.split('.')[-1]) or ('img' in x.split('.')[-1]) or ('JPG' in x.split('.')[-1]) or (
                        'jpg' in x.split('.')[-1]) or ('PNG' in x.split('.')[-1]) or ('png' in x.split('.')[-1]):
                    sol_pics.append(self.sol_dir + x)
                elif ('LBL' in x.split('.')[-1]) or ('lbl' in x.split('.')[-1]) or ('XML' in x.split('.')[-1]) or (
                        'xml' in x.split('.')[-1]):
                    sol_labels.append(self.sol_dir + x)
        except AttributeError as e:
            print(f'Exception {e}')
            pass
        # de-duping just in case
        self.scenes = list(dict.fromkeys(sol_pics))
        self.params = list(dict.fromkeys(sol_labels))


class Frame(object):
    def __init__(self, idx, sol, params_url=None, img_url=None):
        self.idx = idx
        self.sol = sol
        self.params_url = params_url
        self.img_url = img_url

    def download_frame(self, img_directory):
        """This function downloads urls to img_directory"""
        print(self.sol)
        # download IMG and LBL files
        try:
            wget.download(self.params_url, out=img_directory)
        except Exception as ex:
            print(ex)
            print("{} has no label pair".format(self.img_url))
        try:
            wget.download(self.img_url, out=img_directory)
        except Exception as ex:
            print(ex)
            print("{} has no img pair.".format(self.params_url))


def pull_images(sol_start, sol_end, server, directory):
    """This function uses sol and frame class to pull images and their labels to a directory
    """

    # create list of SOLs
    sol_list = []
    for i in range(sol_start, sol_end+1):
        sol_list.append(i)

    if not os.path.exists(directory):
        os.makedirs(directory)

    sol_info = []
    for i in sol_list:
        sol = Sol(i, server)
        sol.parser()
        sol_info.append([sol.idx, sol])

    sol_dict = {}
    for i in range(len(sol_list)):
        sol_dict[sol_info[i][1].label] = sol_info[i][1].__dict__

    for i in sol_dict.keys():
        val1 = len(sol_dict[i]['params'])
        val2 = len(sol_dict[i]['scenes'])
        val3 = max(val1, val2)
        for j in range(val3):
            try:
                params = sol_dict[i]['params'][j]
            except IndexError:
                params = None
            try:
                img = sol_dict[i]['scenes'][j]
            except IndexError:
                img = None
            j_frame = Frame(j, i, params_url=params, img_url=img)
            j_frame.download_frame(directory + '/')

            print("\n sets downloaded: {} of {}".format(j + 1, val3))
