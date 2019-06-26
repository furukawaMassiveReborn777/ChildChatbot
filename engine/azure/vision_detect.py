# coding: utf-8
'''
AZURE PROC 2
object detection using azure api
'''
import requests, json, glob, sys
#import matplotlib.pyplot as plt
from PIL import Image
import settingaz, time

key_dict = json.load(open(settingaz.SECRET_FILE, 'r'))

#SEARCH_WORD = " ".join(settingaz.SEARCH_WORD_LIST)## USEAPP
search_word_underb = sys.argv[1]

SAVE_PAR_DIR = settingaz.SAVE_PAR_DIR
SAVE_DIR = SAVE_PAR_DIR + "/" + search_word_underb


OBJINFO_PATH = settingaz.OBJINFO_PATH

subscription_key = key_dict["vision_key"]
vision_base_url = key_dict["vision_endpoint"]+"vision/v2.0/"# "vision_endpoint" must end with '/'

analyze_url = vision_base_url + "analyze"

image_list = glob.glob(SAVE_DIR+"/*")
image_dict = {}
for image_path in image_list:

    try:
        image_data = open(image_path, "rb").read()
        headers    = {'Ocp-Apim-Subscription-Key': subscription_key,
                      'Content-Type': 'application/octet-stream'}
        params     = {'visualFeatures': 'Objects'}
        response = requests.post(
            analyze_url, headers=headers, params=params, data=image_data)
        response.raise_for_status()

        analysis = response.json()
        image_dict[image_path] = analysis
    except:
        print("error exception")
        time.sleep(1)
    time.sleep(0.5)

json.dump(image_dict, open(OBJINFO_PATH, "w"))