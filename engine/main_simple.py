# coding:utf-8
'''
train and predict for smaller & taller relation
input only box shape infomation(xyhw) to simple multi layer perceptron
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy, time, random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import cv2
import azure.settingaz

TRAIN_MODE = False # False when predicting
TARGET_REL = "tall" # "tall" or "small"  ## USE app
savemodel = "./engine/data/trained_model/simple_target.pth"
if TRAIN_MODE:
    data_path = "./engine/data/procdata/anno_target_rel_dict.pkl"#"./data/anno_small_rel_dict.pkl" #
else: # predict mode for collected images using azure
    data_path = azure.settingaz.AZURE_DICT

NUM_CLASSES = 2 # target or not
BOX_INFO_SIZE = 12 # y1,y2,x1,x2,h,w (x 2box)
TRAIN_EPOCH = 50
VALID_SIZE = 20
BATCH_SIZE = 4

print("----------SETTING----------")
print("TRAIN_MODE={}, savemodel={}, data_path={}".format(TRAIN_MODE, savemodel, data_path))
print("NUM_CLASSES={}, BOX_INFO_SIZE={}, TRAIN_EPOCH={}, VALID_SIZE={}".format(NUM_CLASSES, BOX_INFO_SIZE, TRAIN_EPOCH, VALID_SIZE))
print("---------------------------")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(BOX_INFO_SIZE, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class JudgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, randomize):
        self.data_list = data_list
        self.randomize = randomize

    def __len__(self):
        return len(self.data_list)

    def calc_box_info(self, box, img_height, img_width): # calculate height and weight, then concat infomation
        # box y_start, y_end, x_start, x_end
        y_box = box[:2]/img_height
        x_box = box[-2:]/img_width

        box_h = y_box[1] - y_box[0]
        box_w = x_box[1] - x_box[0]
        hw = np.array([box_h, box_w])
        box_info = np.concatenate([y_box, x_box, hw])
        return box_info

    def __getitem__(self, idx):
        '''
        < example of data format >
        ('imgpath...VG_100K/2317427.jpg',
        {'sub_box_true': array([[  9.,  47., 421., 498.]]),
         'obj_box_true': array([[148., 169., 328., 340.]]),
         'relation_list': [1], 'sub_names': ['balls'], 'obj_names': ['horn']})
        '''
        imgpath = self.data_list[idx][0]
        img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        img_height, img_width, channels = img.shape[:3]

        sub_box = self.data_list[idx][1]['sub_box_true'][0]
        obj_box = self.data_list[idx][1]['obj_box_true'][0]
        sub_box_info = self.calc_box_info(sub_box, img_height, img_width)
        obj_box_info = self.calc_box_info(obj_box, img_height, img_width)

        label = np.array([0, 1], dtype='float32')#taller smaller
        box_info_all = np.concatenate([sub_box_info, obj_box_info])
        if self.randomize:# reverse relation to learn opposite relation
            if random.random() < 0.5:
                label = np.array([1, 0], dtype='float32')#lower bigger
                box_info_all = np.concatenate([obj_box_info, sub_box_info])# opposite input

        box_info_all = box_info_all.astype('float32').reshape(12)
        return box_info_all, label

def train(model, dataloaders, device, num_epoch):
    best_acc = 0.
    for epoch in range(num_epoch):
        for phase in ["train", "valid"]:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            loss_ave = 0.
            correct = 0
            proc_data = 0
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(data)
                    loss = F.mse_loss(output, target)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    output_class = output.argmax(1)
                    target_class = target.argmax(1)
                    correct += (output_class == target_class).sum().item()

                    proc_data += target.size(0)
                    loss_ave += loss.item()

            loss_ave /= proc_data
            epoch_acc = correct / proc_data#(float(VALID_INTERVAL)*4)
            print('Phase:{} Train Epoch: {}  Loss: {:.3f}, Acc: {:.3f}'.format(
                phase, epoch, loss_ave, epoch_acc))

            if phase == "valid" and epoch_acc > best_acc:
                print(">>>best score savemodel")
                best_acc = epoch_acc
                torch.save(model.state_dict(), savemodel)

def predict(model, dataloader, device):
    model.eval()
    pred_np = np.zeros((2))
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            pred_np += output[0].cpu().data.numpy()

        pred_np /= len(dataloader)
        pred_prob = pred_np[1]
    return pred_prob

def run_simple(target_rel, search_words):
    global savemodel, data_path
    savemodel = savemodel.replace("_target", "_"+target_rel)
    data_path = data_path.replace("search_word", search_words)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_net = Net().to(device)

    # load dataset
    with open(data_path, mode='rb') as f:
        data_dict = pickle.load(f)
    data_list = list(data_dict.items())
    print(">data total=", len(data_list))

    judge_dataset = JudgeDataset(data_list=data_list, randomize=TRAIN_MODE)

    if TRAIN_MODE:
        # split dataset
        dataset_size = len(judge_dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        train_indices, valid_indices = indices[:-VALID_SIZE], indices[-VALID_SIZE:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        train_dataloader = DataLoader(judge_dataset, batch_size=BATCH_SIZE,
                                num_workers=2, sampler=train_sampler)
        valid_dataloader = DataLoader(judge_dataset, batch_size=BATCH_SIZE,
                                num_workers=2, sampler=valid_sampler)
        dataloaders_dict = {"train":train_dataloader, "valid":valid_dataloader}
    else:# predict mode
        print(">>>valid data num=", len(data_list))
        if len(data_list) == 0:
            return -1
        pred_dataloader = DataLoader(judge_dataset, batch_size=1,
                                num_workers=1, shuffle=False)
        model_net.load_state_dict(torch.load(savemodel))
        print(">model loaded", savemodel)

    if TRAIN_MODE:
        optimizer = optim.Adam(model_net.parameters(), lr=0.00001)
        print(optimizer)
        train(model_net, dataloaders_dict, device, TRAIN_EPOCH)
    else:
        pred_prob = predict(model_net, pred_dataloader, device)
        print(">pred_prob", pred_prob)
        return pred_prob
    return -2

if __name__ == '__main__':
    search_words_main = "_".join(azure.settingaz.SEARCH_WORD_LIST)
    run_simple(TARGET_REL, search_words_main)