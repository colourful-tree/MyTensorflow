import json
import numpy as np
import re
import itertools
from collections import Counter
import math

def deal_scores(sent, label_score, label_ans):
    sent = sent.replace("[", "")
    sent = sent.replace("]", "")
    sent = sent.replace(" ", "")
    s = [float(i) for i in sent.split(",")]
    cnt = len(label_ans) - len(label_score)
    score = []
    for i in range(cnt):
        score.append(s[i*3:i*3+3][label_ans[i]])
    label_score += score
def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    label_ans = []
    label_score = []
    with open("10820","r") as fin:
        for i in fin:
            if i[:2] == "[[":
                deal_scores(i, label_score, label_ans)
                continue
            if i[0] == "[":
                i = i[1:-2]
                label_ans += [int(i) for i in i.strip().split(",")]
    with open("result.json","w") as fout:
            with open("ccir.test.json", "r") as fin:
                all_data = json.load(fin)
                cnt = 0
                scores = 0
                scores_gold = 0
                for i_data in all_data:
                    p_id = i_data["query_id"]
                    p_query = i_data["query"]
                    p_passage = []
                    passage_list = i_data["passages"]
                    for i_pass in passage_list:
                        url = i_pass["url"]
                        passage_id = i_pass["passage_id"]
                        passage_text = i_pass["passage_text"]
                        label = i_pass["label"]
                        my_label = label_ans[cnt]
                        tmp = {}
                        tmp["url"] = url
                        tmp["passage_id"] = passage_id
                        tmp["passage_text"] = passage_text
                        tmp["label"] = label
                        tmp["my_label"] = label_ans[cnt]
                        tmp["my_label_score"] = label_score[cnt]
                        tmp["test"] = label_ans[cnt] + label_score[cnt]
                        p_passage.append(tmp)
                        cnt += 1
                    #p_passage = sorted(p_passage, key=lambda x: x["my_label"], reverse=True)
                    p_passage = sorted(p_passage, key=lambda x: x["test"], reverse=True)
                    rank = 1
                    for i in p_passage:
                        i["rank"] = rank
                        rank += 1
                    score = 0
                    for i in range(len(p_passage[:3])):
                        score += (2**p_passage[i]["label"] - 1) / math.log(i+2, 2)
                    scores += score
                    p_passage = sorted(p_passage, key=lambda x: x["label"], reverse=True)
                    for i in range(len(p_passage[:3])):
                        scores_gold += (2**p_passage[i]["label"] - 1) / math.log(i+2, 2)
                    if len(p_passage) != 0:
                        tmp = {}
                        tmp["query_id"] = p_id
                        tmp["query"] = p_query
                        tmp["passages"] = p_passage
                        string = json.dumps(tmp, ensure_ascii=False).encode("utf-8")
                        fout.write(string)
                        fout.write(",\n")        
            print scores / 1000
            print scores_gold / 1000
            print scores / scores_gold
            print len(label_score)
            print len(label_ans)
load_data_and_labels("./ccir.json")
