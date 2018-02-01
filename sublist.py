import os
import numpy as np
import h5py
import random
import linecache


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def dir2list(path,sub_list_file):
    if os.path.exists(sub_list_file):
        fp = open(sub_list_file, 'r')
        sublines = fp.readlines()
        sub_names = []
        for subline in sublines:
            sub_info = subline.replace('\n', '')
            sub_names.append(sub_info)
        fp.close()
        return sub_names
    else:
        fp = open(sub_list_file, 'w')
        img_root_dir = os.path.join(path)
        subs = os.listdir(img_root_dir)
        subs.sort()
        for sub in subs:
            sub_dir = os.path.join(img_root_dir,sub)
            views = os.listdir(sub_dir)
            views.sort()
            for view in views:
                view_dir = os.path.join(sub_dir,view)
                slices = os.listdir(view_dir)
                slices.sort()
                for slice in slices:
                    line = os.path.join(view_dir,slice)
                    fp.write(line + "\n")
        fp.close()


def equal_length_two_list(list_A, list_B):
    if len(list_A)<len(list_B):
        diff = len(list_B)-len(list_A)
        for i in range(diff):
            list_A.append(list_B[i])
    else:
        diff = len(list_A)-len(list_B)
        for i in range(diff):
            list_B.append(list_B[i])
    return list_A, list_B