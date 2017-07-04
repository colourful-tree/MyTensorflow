import json
import jieba
import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    """string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)"""
    string = re.sub(r" +"," ", string)
    return string.strip()


def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x_text_1, x_text_2, y, overlap = [], [], [], []
    p_id = []
    with open(data_file, "r") as fin:
        all_data = json.load(fin)
        for i_data in all_data:
            query = i_data["query"]
            query_seg = jieba.cut(query, cut_all=False)
            query = clean_str(" ".join(query_seg))
            passage_list = i_data["passages"]
            id_i = i_data["query_id"]
            # p_id.append(id_i)
            for i_passage in passage_list:
                passage_text = i_passage["passage_text"]
                passage_text_seg = jieba.cut(passage_text, cut_all=False)
                passage_text = clean_str(" ".join(passage_text_seg))
                label = 0#int(i_passage["label"])
                passage_id = i_passage["passage_id"]
                x_text_1.append(query)
                x_text_2.append(passage_text)
                overlap.append(word_over_lap(query, passage_text))
                p_id.append([id_i, passage_id])
                if label == 0:
                    y.append([1,0,0])
                elif label == 1:
                    y.append([0,1,0])
                elif label == 2:
                    y.append([0,0,1])
                else:
                    print "error label"

    #y = np.array(y)
    return [x_text_1, x_text_2, y, overlap, p_id]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
def eva_batch_iter(data, batch_size):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]
def word_over_lap(x, y):
    #print [len(set(x.split(" ")).intersection(set(y.split(" ")))) * 1.0 / len(x.split(" "))]
    return [len(set(x.split(" ")).intersection(set(y.split(" ")))) * 1.0 / len(x.split(" "))]
"""
a,b,c = load_data_and_labels("/root/tensor_word_space/similarity.cnn/my/data/ccir.json")
for i in range(len(a)):
    print a[i].encode("utf-8")
    print b[i].encode("utf-8")
    print c[i]
"""
