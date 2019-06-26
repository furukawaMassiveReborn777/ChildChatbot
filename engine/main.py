# coding:utf-8
'''
train and predict relation
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import os, sys, random, time, pickle
from setting import TRAIN_MODE, MODELPATH, CLASS_SPLIT, IMG_WIDTH, BATCH_SIZE, NUM_CLASSES, GENOME_DICT
from setting import NUM_EPOCHS, VALID_DATA_TOTAL, TRAIN_PROC_MAX, VALID_PROC_MAX, DATA_SHUFFLE, LEARN_VGG, RELATION_FILE, FIRST_CLASS
import azure.settingaz
from model import Net
from dataset import RelDataset

PRED_REL = "eating" ## USE app
#PRED_DATA = azure.settingaz.AZURE_DICT
num_classes = NUM_CLASSES
class_split = CLASS_SPLIT
modelpath = MODELPATH
print("----------SETTING----------")
print("IMG_WIDTH={} BATCH_SIZE={} NUM_EPOCHS={}".format(IMG_WIDTH, BATCH_SIZE, NUM_EPOCHS))
print("TRAIN_PROC_MAX={} DATA_SHUFFLE={} LEARN_VGG={}".format(TRAIN_PROC_MAX, DATA_SHUFFLE, LEARN_VGG))
print("VALID_DATA_TOTAL={} class_split={}".format(VALID_DATA_TOTAL, class_split))
print("---------------------------")

model_dir = "/".join(modelpath.split("/")[:-1])

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

class_names_master = open(RELATION_FILE, "r").read().split("\n")

if class_split == "first":
    class_names = class_names_master[:FIRST_CLASS]
elif class_split == "second":
    class_names = class_names_master[FIRST_CLASS:]

def softmax(arr):
    max_value = np.max(arr)
    exp_np = np.exp(arr - max_value)
    sum_exp_np = np.sum(exp_np, axis=1, keepdims=True)
    soft = exp_np / sum_exp_np
    return soft

# output fixed data size from relation_list
def relation_batch(relation_list, objbox, subbox, batch_size):
    relation_indices = [random.randint(0, len(relation_list)-1) for r in range(batch_size)]
    objbox_batch = np.zeros((batch_size, 4))
    subbox_batch = np.zeros((batch_size, 4))
    labels = torch.zeros([batch_size], dtype=torch.long)
    for idx, rel_idx in enumerate(relation_indices):
        objbox_batch[idx] = objbox[rel_idx]
        subbox_batch[idx] = subbox[rel_idx]
        labels[idx] = relation_list[rel_idx]
    return objbox_batch, subbox_batch, labels


def train_model(model, train_dataset, valid_dataset, optimizer, device):
    model_number = 0
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        # batch_size = 1 because learn size=BATCH_SIZE relations from only one image
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=DATA_SHUFFLE)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=DATA_SHUFFLE)
        dataloaders = {"train":train_dataloader, "valid":valid_dataloader}

        for phase in ['train', 'valid']:
            start = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            total_loss = 0.0
            total_corrects = 0
            total_data = 0
            class_correct = list(0. for i in range(num_classes))
            class_total = list(0. for i in range(num_classes))

            for proc, [inputs, objbox, subbox, relation_list] in enumerate(dataloaders[phase]):
                if len(relation_list) == 0:
                    continue

                if len(objbox.shape) == 3:
                    objbox = torch.squeeze(objbox, dim=0)
                    subbox = torch.squeeze(subbox, dim=0)

                objbox_batch, subbox_batch, labels = relation_batch(relation_list, objbox, subbox, BATCH_SIZE)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    prediction = model(inputs, objbox_batch, subbox_batch)
                    loss = criterion(prediction, labels)
                    _, pred_class = torch.max(prediction, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                total_corrects += torch.sum(pred_class.data == labels.data)
                total_data +=  inputs.size(0) * BATCH_SIZE

                # count correct number of each class
                cor = (pred_class.data == labels.data).squeeze()
                for lidx, label in enumerate(labels):
                    class_correct[label.item()] += cor[lidx].item()
                    class_total[label.item()] += 1

                if proc % 1000 == 0 and phase == "train":
                    print(">proc", proc)
                if (proc+1) % VALID_PROC_MAX == 0 and phase == "valid":
                    break
                if proc >= TRAIN_PROC_MAX:
                    break

            time_elapsed = time.time() - start
            print('time = {:.1f}min'.format(time_elapsed / 60.0))

            epoch_loss = total_loss / total_data
            epoch_acc = total_corrects.double() / total_data
            print('{} corrects:{} Loss: {:.3f} Acc: {:.3f}'.format(phase, total_corrects, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                print(">>>best score, save model_number=", model_number)
                best_acc = epoch_acc
                savemodel_path = modelpath+"_"+str(model_number)
                print(">save model to", savemodel_path)
                torch.save(model.state_dict(), savemodel_path)
                model_number += 1
            if phase == 'valid':
                for n in range(num_classes):
                    if class_total[n] != 0:
                        print("class_correct={} class_total={}".format(class_correct[n], class_total[n]))
                        acc = 100 * class_correct[n] / class_total[n]
                    else:
                        acc = -1.
                    print('class accuracy {} = {:.2f}'.format(n, acc))

    print('Best valid Acc: {:3f}'.format(best_acc))


def pred_relation(pred_target, model, pred_dataset, device):
    pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=1, shuffle=False)

    model.eval()
    pred_class_list = []
    pred_rel_idx = class_names.index(pred_target)
    print(class_names, pred_rel_idx)

    # predict twice to swap box (sub, obj) (obj, sub) and average
    for i in range(2):
        all_pred = torch.zeros([len(pred_dataset), num_classes], dtype=torch.float32)
        for proc, [inputs, objbox, subbox, relation_list] in enumerate(pred_dataloader):
            if len(objbox.shape) == 3:
                objbox = torch.squeeze(objbox, dim=0)
                subbox = torch.squeeze(subbox, dim=0)

            objbox_batch, subbox_batch, labels = relation_batch(relation_list, objbox, subbox, 1)

            if i == 1:
                objbox_batch, subbox_batch = subbox_batch, objbox_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            prediction = model(inputs, objbox_batch, subbox_batch)
            all_pred[proc] = prediction
            _, pred_class = torch.max(prediction, 1)

        all_pred_np = softmax(all_pred.cpu().data.numpy())

        pred_class_ave = np.average(all_pred_np, axis = 0)[pred_rel_idx]
        pred_class_list.append(pred_class_ave)

    # divide positive score by negative score
    pred_ratio = pred_class_list[0] / pred_class_list[1]
    print("pred_ratio=", pred_ratio)
    return pred_ratio

def run(pred_target, search_words):
    global class_names, num_classes, modelpath
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if TRAIN_MODE:
        model = Net(device, BATCH_SIZE, num_classes)
    else:# pred mode
        if pred_target in class_names_master[:FIRST_CLASS]:
            class_names = class_names_master[:FIRST_CLASS]
            modelpath = modelpath.replace("second", "first")
        else:
            class_names = class_names_master[FIRST_CLASS:]
            modelpath = modelpath.replace("first", "second")
        num_classes = len(class_names)
        model = Net(device, 1, num_classes)

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if TRAIN_MODE:
        with open(GENOME_DICT, mode='rb') as f:
            all_data_dict = pickle.load(f)

        all_data_list = list(all_data_dict.items())
        if DATA_SHUFFLE:
            random.shuffle(all_data_list)
        print(">data total=", len(all_data_list))

        train_list = all_data_list[:-VALID_DATA_TOTAL]
        valid_list = all_data_list[-VALID_DATA_TOTAL:]
        train_dataset = RelDataset(data_list=train_list, train_flag=TRAIN_MODE, transform=data_transform)
        valid_dataset = RelDataset(data_list=valid_list, train_flag=TRAIN_MODE, transform=data_transform)

    if LEARN_VGG:
        params_to_update = model.parameters()
    else:
        params_to_update = []
        for name, param in model.named_parameters():
            if "vgg" in  name:
                param.requires_grad = False
            else:
                params_to_update.append(param)


    model = model.to(device)

    if TRAIN_MODE:
        print("-----train mode")
        optimizer_ft = optim.Adam(params_to_update, lr=0.00001)
        print(optimizer_ft)
        train_model(model, train_dataset, valid_dataset, optimizer_ft, device)
    else:
        PRED_DATA = azure.settingaz.AZURE_DICT.replace("search_word", search_words)
        print("-----predict mode, load model from", modelpath)
        with open(PRED_DATA, mode='rb') as f:
            all_pred_dict = pickle.load(f)
        print(">>>valid data num=", len(all_pred_dict))
        if len(all_pred_dict) == 0:
            return -1

        all_pred_list = list(all_pred_dict.items())
        pred_dataset = RelDataset(data_list=all_pred_list, train_flag=TRAIN_MODE, transform=data_transform)
        model.load_state_dict(torch.load(modelpath))
        pred_ratio = pred_relation(pred_target, model, pred_dataset, device)
        return pred_ratio

    return -2

if __name__ == '__main__':
    search_words_main = "_".join(azure.settingaz.SEARCH_WORD_LIST)
    run(PRED_REL, search_words_main)
