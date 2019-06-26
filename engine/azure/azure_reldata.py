# coding:utf-8
'''
AZURE PROC3
convert azure object detect result to
predictable format data
'''
import numpy as np
import json, os
import pickle
import cv2, sys
import time
import copy, random
import settingaz
sys.path.append(os.getcwd())
from engine.preproc.synonym import find_synonym


OBJINFO_PATH = settingaz.OBJINFO_PATH
#SEARCH_WORD_LIST = settingaz.SEARCH_WORD_LIST## USEAPP
#SEARCH_WORD = " ".join(settingaz.SEARCH_WORD_LIST)## USEAPP
search_word_origin = sys.argv[1]
savepath = settingaz.AZURE_DICT.replace("search_word", search_word_origin)
search_word_list = search_word_origin.split("_")
search_word_list.pop(1)
print(search_word_list, savepath)

search_word_syn1 = find_synonym(search_word_list[0])
search_word_syn2 = find_synonym(search_word_list[1])

all_info = json.load(open(OBJINFO_PATH, "r"))

all_data_dict = {}
for imgpath, image_info in all_info.items():
    # example == 'imgpath', {'sub_box_true': array([[488., 536.,  22., 121.]]), 'obj_box_true': array([[368., 599.,   0., 472.]]), 'relation_list': [-1], 'sub_names': ['cat'], 'obj_names': ['dog']}
    info_dict = {'relation_list':[-1]}
    objects = image_info["objects"]
    for obj in objects:
        obj_list = []
        box = None
        for k, v in obj.items():
            if "rectangle" in k:
                # y_start, y_end, x_start, x_end
                box = np.array([[v['y'], v['y']+v['h'], v['x'], v['x']+v['w']]])
            elif "object" in k:
                obj_list.extend(v.split(" "))
            elif "parent" in k:
                obj_list.extend(v["object"].split(" "))
                while True:
                    if "parent" in v:
                        v = v["parent"]
                        obj_list.extend(v["object"].split(" "))
                    else:
                        break

        obj_list = [find_synonym(obj) for obj in obj_list]
        #print(">>obj_list", obj_list)
        if search_word_syn1 in obj_list:
            info_dict['sub_box_true'] = box
            info_dict['sub_names'] = search_word_list[0]
        elif search_word_syn2 in obj_list:
            info_dict['obj_box_true'] = box
            info_dict['obj_names'] = search_word_list[1]

    if len(info_dict) == 5:# use image with both sub and obj
        all_data_dict[imgpath] = info_dict
        print(">valid img=", imgpath)

print("valid images total=", len(all_data_dict))
print("save to", savepath)
with open(savepath, mode='wb') as f:
    pickle.dump(all_data_dict, f)
