# coding: utf-8
'''
AZURE PROC 1
image search and save using azure api
'''
import subprocess
from requests import exceptions
import requests
import cv2
import os, sys
import settingaz
import json

if not os.path.exists(settingaz.SAVE_DIR):
    os.makedirs(settingaz.SAVE_DIR)

#SEARCH_WORD = " ".join(settingaz.SEARCH_WORD_LIST)## USEAPP
search_word_underb = sys.argv[1]#SEARCH_WORD.replace(" ", "_")
search_word_use = search_word_underb.replace("_", " ")

key_dict = json.load(open(settingaz.SECRET_FILE, 'r'))

MAX_RESULTS = 2
COUNTS = 20

MAX_FILE_SIZE = 10000000#10MB
subscription_key = key_dict["search_key"]

URL = key_dict["search_endpoint"]+"/images/search"
SAVE_PAR_DIR = settingaz.SAVE_PAR_DIR
SAVE_DIR = SAVE_PAR_DIR + "/" + search_word_underb
if not os.path.exists(SAVE_PAR_DIR):
    os.mkdir(SAVE_PAR_DIR)
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

EXCEPTIONS = set([IOError, FileNotFoundError,
    exceptions.RequestException, exceptions.HTTPError,
    exceptions.ConnectionError, exceptions.Timeout])


headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
params = {"q": search_word_use, "offset": 0, "count": COUNTS, "imageType":"Photo", "color":"ColorOnly"}

print("search_word_use={}".format(search_word_use))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

results = search.json()
result_total = min(results["totalEstimatedMatches"], MAX_RESULTS)

total = 0

for offset in range(0, result_total, COUNTS):
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("images {}-{} of {}...".format(
        offset, offset + COUNTS, result_total))

    for v in results["value"]:
        try:
            r = requests.get(v["contentUrl"], timeout=10)

            ext = v["contentUrl"][v["contentUrl"].rfind("."):v["contentUrl"].rfind("?") if v["contentUrl"].rfind("?") > 0 else None]
            p = os.path.sep.join([SAVE_DIR, "{}{}".format(
                str(total).zfill(2), ext)])

            if type(r.content) == bytes and len(r.content) < MAX_FILE_SIZE:
                f = open(p, "wb")
                f.write(r.content)
                f.close()

        except Exception as e:
            if type(e) in EXCEPTIONS:
                print("skip url= {}".format(v["contentUrl"]))
                continue

        image = cv2.imread(p)
        if image is None:
            print("delete non valid file:{}".format(p))
            if os.path.exists(p):
                os.remove(p)
            continue
        total += 1

print(">image search done")