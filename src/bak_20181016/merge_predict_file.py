import os
import numpy as np
import tensorflow as tf
import re
tf.app.flags.DEFINE_string('predict_output_file_dir', "/home/niusilong/work/AI/predict", """最低预测分数""")
tf.app.flags.DEFINE_string('predict_output_file_name', "predict_result", """""")
FLAGS = tf.app.flags.FLAGS
files = os.listdir(FLAGS.predict_output_file_dir)
predict_files = []
for i in range(len(files)):
    if files[i].startswith(FLAGS.predict_output_file_name):
        predict_files.append(files[i])
predict_dict = {}

for file in predict_files:
    file_path = os.path.join(FLAGS.predict_output_file_dir, file)
    with open(file_path) as f:
        while True:
            line = f.readline().strip()
            if line == "":
                break
            line_array = line.split(" ")
            img = line_array[0]
            labels = line_array[1].split(",")
            if not predict_dict.keys().__contains__(img):
                predict_dict[img] = {}
            for i in range(len(labels)):
                predict_dict[img][labels[i]] = 1 if not predict_dict[img].keys().__contains__(labels[i]) else predict_dict[img][labels[i]]+1
img_sorted_keys = sorted(predict_dict.keys())
print(img_sorted_keys)
merged_dict = {}
min_num = int(len(predict_files)/10)
print("min_num:",min_num)
for key in img_sorted_keys:
    # print(key,predict_dict[key])
    merged_dict[key] = []
    keys = np.asarray(list(predict_dict[key].keys()))
    values = np.asarray(list(predict_dict[key].values()))
    sorted_array = values.argsort()[::-1]
    # for i in range(len(sorted_array)):
    #     print("\tlabel:%s, count:%d" % (keys[sorted_array[i]], values[sorted_array[i]]))
    # print("total:",total, "average:", average, "min_num:",min_num)
    for i in range(len(sorted_array)):
        if values[sorted_array[i]] > min_num:
            merged_dict[key].append(keys[sorted_array[i]])
        # print("\tlabel:%s, count:%d" % (keys[sorted_array[i]], values[sorted_array[i]]))
for key in img_sorted_keys:
    print(key,merged_dict[key])
#export file
with open(os.path.join(FLAGS.predict_output_file_dir, "merged_"+FLAGS.predict_output_file_name+".txt"), "w") as f:
    for key in img_sorted_keys:
        f.write(key+" "+re.sub("['\\[\\]\\s]","",str(merged_dict[key]))+"\n")