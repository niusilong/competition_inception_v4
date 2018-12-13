'''
合并切图预测的标签
'''
import os
import re
import numpy as np
DEFAULT_PREDICT_IMG_DIR = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/webapp/images/120_4"
DEFAULT_PREDICT_FILE_DIR = "/home/niusilong/work/AI/predict"
# FILE_PREFIX = "crop4_predict_result_"
FILE_PREFIX = "score_full_predict_result_"
DEFAULT_EVALUATE_FILE_LATEST_COUNT = 100
MAX_LABEL_COUNT = 8
PRINT_SCORE = True
MERGED_PREDICT_FILE_PREFIX = "merged_"
class ImageLabel(object):
    def __init__(self, image, labels):
        self.image = image
        self.labels = labels
class LabelScore(object):
    def __init__(self, label, score):
        self.label = label
        self.score = score
    def __repr__(self):
        return self.label+","+str(self.score)+","+str(self.softmax_score)+"(repr)"
    def __str__(self):
        return self.label+str(self.score)

def merge(src_image_dir, lines):
    original_parent_array = []
    parent_images_list = os.listdir(src_image_dir)
    for image in parent_images_list:
        original_parent_array.append(int(image[:image.find(".")]))
    original_parent_array.sort()
    crop4_array = []
    for line in lines:
        line_array = line.split(" ")
        image = line_array[0]
        image_label = ImageLabel(image, line_array[1].split(",") if len(line_array) == 2 else [])
        crop4_array.append(image_label)
    output_array = []
    for o_image in original_parent_array:
        sum_labels = []
        for image_label in crop4_array:
            if image_label.image.startswith(str(o_image)+"_"):
                labels = image_label.labels
                for label in labels:
                    do_max_label(sum_labels, label[:label.find("(")], float(label[label.find("(")+1:label.find(")")]))
        sum_labels.sort(key=label_score_sort_key, reverse=True)
        sum_scores = []
        for label_score in sum_labels:
            sum_scores.append(label_score.score)
        sum_scores = softmax(sum_scores)
        for i in range(len(sum_scores)):
            sum_labels[i].softmax_score = sum_scores[i]
        # print("sum_labels:",sum_labels)
        output_labels = []
        for i in range(len(sum_labels)):
            output_labels.append(sum_labels[i].label+(("("+str(sum_labels[i].score)+")") if PRINT_SCORE else ""))
            if len(output_labels) >= MAX_LABEL_COUNT:
                print("sum_labels:",sum_labels[:i])
                break

        # [label_score.label for label_score in sum_labels]
        output_array.append(str(o_image)+" "+re.sub("['\\[\\]\\s]","",str(output_labels)))
    return output_array
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)
def label_score_sort_key(item):
    return item.score
def do_max_label(sum_labels, label, score):
    # print("sum_labels:",sum_labels)
    contains = False
    for label_score in sum_labels:
        if label == label_score.label:
            if score >= label_score.score:
                label_score.score = score
            # label_score.score += score
            contains = True
            break
    if not contains:
        sum_labels.append(LabelScore(label, score))
def sort_by_time(filepath):
    return os.path.getmtime(filepath)
def write_merged_files(src_image_dir, predict_file_dir, evaluate_latest_file_count):
    files = os.listdir(predict_file_dir)
    evaluate_files = []
    for file in files:
        if file.startswith(FILE_PREFIX):
            evaluate_files.append(os.path.join(predict_file_dir, file))
    evaluate_files = sorted(evaluate_files, key=sort_by_time)
    print(evaluate_files[0:evaluate_latest_file_count])
    for file in evaluate_files[0:evaluate_latest_file_count]:
        print("merge file:",file)
        with open(file) as f:
            lines = f.readlines()
            print(lines[0])
            merged_lines = merge(src_image_dir, lines)
        with open(os.path.join(predict_file_dir, ("score_" if PRINT_SCORE else "")+MERGED_PREDICT_FILE_PREFIX+file.split("/")[-1]), "w") as f:
            for line in merged_lines:
                f.write(line+"\n")

if __name__ == '__main__':
    # files = os.listdir(DEFAULT_PREDICT_FILE_DIR)
    # evaluate_files = []
    # for file in files:
    #     if file.startswith(FILE_PREFIX):
    #         evaluate_files.append(os.path.join(DEFAULT_PREDICT_FILE_DIR, file))
    # evaluate_files = sorted(evaluate_files, key=sort_by_time, reverse=True)
    # print(evaluate_files[0:EVALUATE_FILE_LATEST_COUNT])
    # for file in evaluate_files[0:EVALUATE_FILE_LATEST_COUNT]:
    #     print("merge file:",file)
    #     with open(file) as f:
    #         lines = f.readlines()
    #         print(lines[0])
    #         merged_lines = merge(lines)
    #     with open(os.path.join(DEFAULT_PREDICT_FILE_DIR, MERGED_PREDICT_FILE_PREFIX+file.split("/")[-1]), "w") as f:
    #         for line in merged_lines:
    #             f.write(line+"\n")
    write_merged_files(DEFAULT_PREDICT_IMG_DIR, DEFAULT_PREDICT_FILE_DIR, DEFAULT_EVALUATE_FILE_LATEST_COUNT)