TRAIN_MODE = False
MODELPATH = './engine/data/trained_model/rel_first_model.pth'
NUM_EPOCHS = 10000
BATCH_SIZE = 30

RELATION_FILE = "./engine/data/refer/relation_train.txt"
GENOME_DICT = "./engine/data/procdata/genome_rel_dict.pkl"
IMG_WIDTH = 800
TRAIN_PROC_MAX = 10000
VALID_PROC_MAX = 1000
DATA_SHUFFLE = True
LEARN_VGG = False

CLASS_SPLIT = "first" # "none", "first", "second" choose class to train
FIRST_CLASS = 3
VALID_DATA_TOTAL = 500
if CLASS_SPLIT == "first":
    NUM_CLASSES = FIRST_CLASS
    VALID_DATA_TOTAL = 300
elif CLASS_SPLIT == "second":
    NUM_CLASSES = 6
