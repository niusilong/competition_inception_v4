'''
合并带分数的预测文件,统计所有得分超过0.5的标签
'''
import re
import os
from file_process import class_image_label_score
SCORE_PREDICT_FILE_DIR = "/home/niusilong/work/AI/predict"
OUTPUT_PREDICT_FILE = "/home/niusilong/work/AI/predict/merged_all_score_full_predict_result.txt"
# SCORE_PREDICT_FILE_PREFIX = "score_merged_score_full_predict_result_"   #切图预测
SCORE_PREDICT_FILE_PREFIX = "score_full_predict_result_"   #全图预测
MERGE_FILE_COUNT = 50
MAX_LABEL_COUNT = 6
'''
class ImageLabel(object):
    def __init__(self, image, label_scores):
        self.image = image
        self.type = type
        label_score_array = []
        # print("label_scores:",label_scores)
        for label_score in label_scores:
            score = float(label_score[label_score.find("(")+1:label_score.find(")")])
            label_score_array.append(LabelScore(label_score[:label_score.find("(")], score))
        self.label_scores = label_score_array
    def __repr__(self):
        return self.image+" "+str(self.label_scores)
class LabelScore(object):
    def __init__(self, label, score):
        self.label = label
        self.score = score
    def __repr__(self):
        return self.label+"("+str(self.score)+")"
    def __str__(self):
        return self.label+"("+str(self.score)+")"
'''
def sort_by_time(filepath):
    return os.path.getmtime(filepath)
def get_predict_file_info(predict_file):
    '''
    :param predict_file:
    :param type: full-全图预测，crop-切图预测
    :return:
    '''
    predict_file_infos = []
    with open(predict_file) as f:
        while True:
            line = f.readline().strip()
            if line == "": break
            line_array = line.split(" ")
            predict_file_infos.append(class_image_label_score.ImageLabel(line_array[0], line_array[1].split(",")))
    return predict_file_infos
def merge_predict_file_infos(predict_file_info_1, predict_file_info_2):
    merged_predict_info = predict_file_info_1
    for i in range(len(merged_predict_info)):
        for image_label_2 in predict_file_info_2:
            if merged_predict_info[i].image == image_label_2.image:
                merged_label_scores = max_label_scores(merged_predict_info[i].label_scores, image_label_2.label_scores)
                merged_predict_info[i].label_scores = merged_label_scores
    return merged_predict_info
def sort_by_score(label_score):
    return label_score.score
def sort_img(filename):
    index = filename.split(".")[0].split("_")[-1]
    return int(index)
def max_label_scores(label_scores_1, label_scores_2):
    max_label_scores = label_scores_1
    # print("label_scores_1:",label_scores_1)
    # print("label_scores_2:",label_scores_2)
    for label_score_2 in label_scores_2:
        contains = False
        for max_label_score in max_label_scores:
            if max_label_score.label == label_score_2.label:
                contains = True
                if label_score_2.score > max_label_score.score:
                    max_label_score.score = label_score_2.score
                break
        if not contains:
            max_label_scores.append(label_score_2)
    max_label_scores = sorted(max_label_scores, key=sort_by_score, reverse=True)
    # print("max_label_scores:",max_label_scores)
    return max_label_scores
if __name__ == '__main__':
    all_files = os.listdir(SCORE_PREDICT_FILE_DIR)
    score_predict_files = []
    for filename in all_files:
        if filename.startswith(SCORE_PREDICT_FILE_PREFIX):
            score_predict_files.append(filename)
    score_predict_files = sorted(score_predict_files, key=sort_img)
    score_predict_files = score_predict_files[len(score_predict_files)-MERGE_FILE_COUNT:]
    print("score_predict_files:",score_predict_files)
    merged_predict_file_info = get_predict_file_info(os.path.join(SCORE_PREDICT_FILE_DIR, score_predict_files[0]))
    print("merged_predict_file_info:",merged_predict_file_info)
    for i in range(len(score_predict_files))[1:]:
        predict_file_info = get_predict_file_info(os.path.join(SCORE_PREDICT_FILE_DIR, score_predict_files[i]))
        merged_predict_file_infos = merge_predict_file_infos(merged_predict_file_info, predict_file_info)
    with open(OUTPUT_PREDICT_FILE, 'w') as f:
        for image_label in merged_predict_file_infos:
            f.write(image_label.image+" "+re.sub("['\\[\\]\\s]","",str(image_label.label_scores))+"\n")