import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(users_file, movies_file, ratings_file):
    """
    """
    user = {}
    movies = {}
    x_text_1 = []
    x_text_2 = []
    y = []
    with open(users_file, "r") as fin:
        for i in fin:
            l = i.strip().split("::")
            user[l[0]] = l[:4]
    with open(movies_file, "r") as fin:
        for i in fin:
            l = i.strip().split("::")
            cla = l[2].split("|")
            movies[l[0]] = l[:2]
            movies[l[0]].append(" ".join(cla))
    with open(ratings_file, "r") as fin:
        for i in fin:
            l = i.strip().split("::")
            if l[0] in user and l[1] in movies:
                x_text_1.append(user[l[0]])
                x_text_2.append(movies[l[1]])
                tmp = []
                tmp.append(int(l[2]))
                y.append(tmp)
    #print x_text_1[0]
    #print x_text_2[0]
    """x_text_1 = np.array(x_text_1)
    x_text_2 = np.array(x_text_2)
    y = np.array(y)"""
    return [x_text_1, x_text_2, y]
    


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
