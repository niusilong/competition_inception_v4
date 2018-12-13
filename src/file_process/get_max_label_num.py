import numpy as np
import os
# IMAGE_LABEL_FILE="/home/niusilong/work/AI/training/train_tagId.txt"
IMAGE_LABEL_FILE="/home/niusilong/svn/svn_repository_ai/competition_inception_v4/predict_2/lat_0927_3.txt"
# IMAGE_LABEL_FILE="/home/niusilong/work/AI/predict/standard_competition_besttone.txt"
max_num = 0
num_array = np.zeros([5], dtype=np.int32)
image_array_sorted_by_num = [[],[],[],[],[]]
print("num_array:",num_array)
with open(IMAGE_LABEL_FILE) as f:
    while True:
        line = f.readline().strip()
        if line == "":
            break
        line_array = line.split(" ")
        if len(line_array) == 1:
            continue
        label_array = line_array[1].split(",")
        label_count = len(label_array)
        if label_count > max_num:
            max_num = label_count
        num_array[label_count-1] += 1
        image_array_sorted_by_num[label_count-1].append(line_array[0])
print("num_array:",num_array)
print("max_num:",max_num)
print(image_array_sorted_by_num[4])