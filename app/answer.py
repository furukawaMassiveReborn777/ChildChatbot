# coding:utf-8
'''
answer to question
bot engine module
'''
import math
import nltk
import re, random
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer as PS
from nltk.corpus import wordnet as wn
import subprocess, sys, os
sys.path.append("./engine")
sys.path.append(os.getcwd())
from engine.azure import settingaz
from engine.preproc.synonym import find_synonym
#nltk.download('punkt')# only first time
#nltk.download('averaged_perceptron_tagger')

ALPHA_REG = re.compile('[^a-zA-Z]')
COMPARE_WORDS = ["smaller", "bigger", "taller", "lower"]
COMPARE_OP_DICT = {"bigger":"smaller", "lower":"taller"}
VERB_LIST_ORI = ["drive", "sleep", "drink", "grow", "ride", "eat", "fly", "wear", "carry"]
PRED_DICT = {"drive":"driving", "sleep":"sleeping-on",
        "drink":"drinking", "grow":"growing-in", "ride":"riding",
        "eat":"eating", "fly":"flying-in", "wear":"wearing", "carry":"carrying"}
REVISE_DICT = {"flies":"fly", "carries":"carry"}

verb_list = [v + "s" for v in VERB_LIST_ORI]
verb_list.extend(VERB_LIST_ORI)

ANSWER_CAND_PATH = "./engine/data/refer/answer_cand.txt"
VIDEO_VECTOR_PATH = "./engine/data/refer/allucfhm1_ave.npy"
VIDEO_CLASSNAME_PATH = "./engine/data/refer/video_train.txt"

STOP_WORDS = ["a", "the", "and", "or"]

all_nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
ps = PS()

def extract_target(text):
    text = text.lower()
    word_list = word_tokenize(text)
    pos_list = nltk.pos_tag(word_list)

    noun_list = []
    target_word = None
    verb = None
    hear_flag = False
    compare_flag = False
    if "you hear" in text:
        hear_flag = True

    for w_ori, p in zip(word_list, pos_list):
        if w_ori in STOP_WORDS:
            continue
        if w_ori in REVISE_DICT:
            w_ori = REVISE_DICT[w_ori]

        w_ori = ALPHA_REG.sub('', w_ori)
        w_ori = find_synonym(w_ori)
        if len(w_ori) == 0:
            continue
        w = ps.stem(w_ori)
        pos = p[1]

        if w_ori in all_nouns and w not in COMPARE_WORDS:
            if w_ori.endswith('s'):
                w_ori = w_ori[:-1]
            noun_list.append(w_ori)
        elif "VB" in pos:
            verb = w_ori

        if w in COMPARE_WORDS:
            target_word = w
            compare_flag = True
        elif w_ori in verb_list and not compare_flag:
            w = w_ori.rstrip("s")
            target_word = w

    if target_word is None:
        target_word = "none"

    target_word = target_word.rstrip("s")
    if target_word in PRED_DICT:
        target_word = PRED_DICT[target_word]
    return target_word, noun_list, verb, hear_flag, compare_flag


def video_vector_mode(question):
    import ngram
    import numpy as np
    import sklearn
    from scipy import spatial
    input_text = question.lower().split("you hear ")[-1]
    vector_np = np.load(VIDEO_VECTOR_PATH)
    classname_list = open(VIDEO_CLASSNAME_PATH, "r").read().split("\n")
    max_sim = 0.
    max_idx = 0
    for i, c in enumerate(classname_list):
        similarity = ngram.NGram.compare(input_text, c)
        if similarity >= max_sim:
            max_idx = i
            max_sim = similarity

    target_vector = vector_np[max_idx]
    dist_list = []
    for i, v in enumerate(vector_np):
        similarity = ngram.NGram.compare(input_text, c)
        if similarity >= max_sim:
            max_idx = i
            max_sim = similarity
        dist = spatial.distance.cosine(target_vector, v)
        if i == max_idx:
            dist = 1.0
        dist_list.append(dist)
    dist_indices = np.argsort(np.array(dist_list))
    c1 = classname_list[dist_indices[0]].replace("_", " ")
    c2 = classname_list[dist_indices[1]].replace("_", " ")
    return c1 + " and " + c2


def judge_class(question):
    output_class = -2 # -2=can not understand, -1=no valid image, 0=no strongly, 1=no, 2=yes, 3=yes strongly
    video_mode_answer = "none"

    target_word, noun_list, verb, hear_flag, compare_flag = extract_target(question)
    print(">extracted", target_word, noun_list)#, verb, hear_flag, compare_flag

    if hear_flag:# video memory concept mode
        video_mode_answer = video_vector_mode(question)
        output_class = 100
        return output_class, video_mode_answer

    if target_word == "none":
        print("---can not understand Q target none")
        return output_class, video_mode_answer

    # image memory concept mode
    compare_op_flag = False
    if target_word in COMPARE_OP_DICT:
        target_word = COMPARE_OP_DICT[target_word]
        compare_op_flag = True

    if compare_flag:
        noun_list.insert(1, "and")
    else:
        noun_list.insert(1, target_word)

    if len(noun_list) == 4:
        print(">noun_list length 4")
        noun_list.pop(2)
    if len(noun_list) != 3:
        print("---can not understand Q", noun_list)
        return output_class, video_mode_answer

    search_words = "_".join(noun_list)
    info_savepath = settingaz.AZURE_DICT.replace("search_word", search_words)

    # if there is already collected infomation, use it
    if not os.path.exists(info_savepath):
        # collect image infomation using azure
        print(">phase1: search")
        p = subprocess.Popen(("python engine/azure/image_search.py "+search_words).split())
        p.wait()
        print(">phase2: obj detect")
        p = subprocess.Popen(("python engine/azure/vision_detect.py "+search_words).split())
        p.wait()
        print(">phase3: proc data")
        p = subprocess.Popen(("python engine/azure/azure_reldata.py "+search_words).split())
        p.wait()

    if compare_flag:# simple mode
        print("---simple mode")
        from engine.main_simple import run_simple
        pred_ratio = run_simple(target_word.rstrip("er"), search_words)
        if pred_ratio == -1:
            output_class = -1
            return output_class, video_mode_answer
        if compare_op_flag:
            pred_ratio = 1. - pred_ratio
        # convert pred_ratio to class
        pred_ratio -= 0.46
        pred_ratio /= 0.02
        if pred_ratio < 0:
            output_class = 0
        elif pred_ratio > 3:
            output_class = 3
        else:
            output_class = math.floor(pred_ratio)
    else:
        print("---relation mode")
        from engine.main import run
        pred_ratio = run(target_word, search_words)
        if pred_ratio == -1:
            output_class = -1
            return output_class, video_mode_answer
        # convert pred_ratio to class
        if pred_ratio < 0.01:
            output_class = 0
        elif pred_ratio < 1.0:
            output_class = 1
        elif pred_ratio < 500:
            output_class = 2
        else:
            output_class = 3
    return output_class, video_mode_answer


def answer(question):
    output_class, video_mode_answer = judge_class(question)
    if output_class == 100:
        return video_mode_answer
    else:
        answer_list = open(ANSWER_CAND_PATH, "r").read().split("\n")
        for ai, a in enumerate(answer_list):
            answer_list[ai] = a.split("_")
        answer_pos = output_class + 2
        answer_output = random.choice(answer_list[answer_pos])
        return answer_output


if __name__ == '__main__':
    Q = "what do you imagine when you hear basketball?"#"a cat is bigger than a dog?"
    if len(sys.argv) == 2:
        Q = sys.argv[1].replace("_", " ")
    A = answer(Q)
    print(">>>QA=", Q, A)
