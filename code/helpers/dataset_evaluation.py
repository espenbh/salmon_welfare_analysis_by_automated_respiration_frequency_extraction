from helpers import m
import numpy as np

def print_class_distribution(data_loader):
    cls_cnt = {x:0 for x in range(0, m.NUM_CLASSES_T9)}
    for images, labels in data_loader:
        for label in labels:
            lab = int(label.detach().cpu().numpy())
            cls_cnt[lab] = cls_cnt[lab] + 1
    for key in cls_cnt.keys():
        print('Class: ', key, ' has ', cls_cnt[key], 'datapoints')


def size_smallest_class(data_loader):
    cls_cnt = {x:0 for x in range(0, m.NUM_CLASSES_T9)}
    for images, labels in data_loader:
        for label in labels:
            lab = int(label.detach().cpu().numpy())
            cls_cnt[lab] = cls_cnt[lab] + 1
    cls_cnt.pop(0)
    cls_cnt.pop(13)
    cls_cnt.pop(14)
    return np.min(np.array(list(cls_cnt.values())))