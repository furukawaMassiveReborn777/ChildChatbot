# coding:utf-8
'''
類義語を見つけて変換する
'''
SYN_PATH = "./engine/data/refer/synonyms.txt"

def find_synonym(search_word):
    search_word = search_word.lower()
    word_str = open(SYN_PATH).read()
    if search_word not in word_str:
        return search_word
    synonym = search_word
    words = word_str.split("\n")
    words = [w.split(",") for w in words]
    for w_list in words:
        if search_word in w_list:
            synonym = w_list[0]
            break
    return synonym

if __name__=='__main__':
    syn = find_synonym("frog")
    print(syn)