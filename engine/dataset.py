# coding:utf-8
'''
dataset class:train and predict relation
'''
import torch
import cv2
import copy
import setting

'''DATA preproc'''
class RelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, train_flag, transform=None):
        self.data_list = data_list
        self.train_flag = train_flag
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_tuple = self.data_list[idx]
        imgpath = img_tuple[0]
        img_info = img_tuple[1]

        ## caution : replace imgpath with your own visual genome directory path
        image = cv2.imread(imgpath)
        resize_scale = setting.IMG_WIDTH/float(image.shape[1])
        img_height = int(image.shape[0] * resize_scale)
        image = cv2.resize(image, (setting.IMG_WIDTH, img_height))
        objbox = img_info["obj_box_true"] * resize_scale#box == y_start, y_end, x_start, x_end
        subbox = img_info["sub_box_true"] * resize_scale

        relation_list = img_info["relation_list"]

        if self.train_flag:
            new_relation_list = []
            new_objbox = copy.deepcopy(objbox)
            new_subbox = copy.deepcopy(subbox)
            for r, rel in enumerate(relation_list):
                if setting.CLASS_SPLIT == "first" and rel < setting.NUM_CLASSES:
                    new_rel = rel
                elif setting.CLASS_SPLIT == "second" and rel >= setting.FIRST_CLASS:
                    new_rel = rel - setting.FIRST_CLASS
                else:
                    continue
                new_objbox[len(new_relation_list)] = objbox[r]
                new_subbox[len(new_relation_list)] = subbox[r]
                new_relation_list.append(new_rel)
            if len(new_relation_list) > 0:
                objbox = new_objbox[:len(new_relation_list)]
                subbox = new_subbox[:len(new_relation_list)]
            relation_list = new_relation_list

        if self.transform:
            image = self.transform(image)

        return image, objbox, subbox, relation_list