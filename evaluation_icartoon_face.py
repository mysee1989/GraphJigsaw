import os
import sys
import math
import numpy as np
import heapq
from threading import Thread
from datetime import datetime as dt
import argparse
import pdb


def get_label_id_dict(labels):
    '''
    feat_file_path : the path of extracted feature file
    id_pos: the position of id in image path which is the line.split(' ')[0]

    return 
       id_line: the line that certain id lies
       line_id: the id of each line
       line_path: the path of image
    '''
    id_line = {}
    line_id = {}
    line_path = {}

    for num,line in enumerate(labels):
        line = int(line)
        if line not in id_line:
            id_line[line] = set()
        id_line[line].add(num)
        line_id[num] = line
    return id_line, line_id

def evaluation(features,labels,total_rank = 10):

    id_line, line_id = get_label_id_dict(labels)
    distractor_feats = features[list(id_line[-1])]
    corroct = [0] *total_rank
    total = 0
    for i in range(len(features)):
        if i%500 == 0:
            print("{}/{} have finished!!!".format(i,len(features)))
        if line_id[i] == -1:
            continue
        gallery_feats = np.concatenate((distractor_feats,np.expand_dims(features[i], axis=0)),axis=0)
        index = id_line[line_id[i]]
        index.remove(i)
        prob_feats = features[list(index)]
        sims = prob_feats.dot(gallery_feats.T)
        # print(sims.shape)

        for j in range(sims.shape[0]):
            sort_index = np.argsort(sims[j])
            for k in range(total_rank):
                if (sims.shape[1]-1) in sort_index[-1-k:]:
                    corroct[k] +=1
        total += sims.shape[0]
    rank_n = []
    for i in range(total_rank):
        rank_n.append(float(corroct[i])/total)

    return rank_n



