'''
合并带分数的全图预测文件和切图预测文件
'''
import re
import os
from file_process.strategy import merge_label_score_util
from file_process import class_image_label_score
# SCORE_PREDICT_FILE_DIR = "/home/niusilong/svn/svn_repository_ai/competition_inception_v4/predict_3"
SCORE_PREDICT_FILE_DIR = "/home/niusilong/work/AI/predict/high_score"

'''sigmoid分数预测'''
#全图预测
# MIN_PREDICT_SCORE = 0.15  #不超出数量限制的最低分数 default=0.4
# EXCEED_MIN_SCORE = 0.5   #超出数量限制依然有效的分数, default=0.5
# SCORE_PREDICT_FILE_PREFIX = "score_full_predict_result_"

#切图预测
# MIN_PREDICT_SCORE = 0.7  #不超出数量限制的最低分数
# EXCEED_MIN_SCORE = 0.9   #超出数量限制依然有效的分数
# SCORE_PREDICT_FILE_PREFIX = "score_merged_score_full_predict_result_"

#全图和切图综合预测
MIN_PREDICT_SCORE = 0.85  #不超出数量限制的最低分数,default=0.85
EXCEED_MIN_SCORE = 0.9   #超出数量限制依然有效的分数, default=0.9
SCORE_PREDICT_FILE_PREFIX = "merged_full_and_cropped_predict_file_score_"   #多个文件合并

#单个文件预测(仅合并全图预测)
# MIN_PREDICT_SCORE = 0.5  #不超出数量限制的最低分数
# EXCEED_MIN_SCORE = 0.9   #超出数量限制依然有效的分数
# SCORE_PREDICT_FILE_PREFIX = "merged_all_score_full_predict_result.txt"   #单个文件合并


#单个文件预测(合并全图预测和切图预测)
# MIN_PREDICT_SCORE = 0.7  #不超出数量限制的最低分数
# EXCEED_MIN_SCORE = 0.9   #超出数量限制依然有效的分数
# SCORE_PREDICT_FILE_PREFIX = "merged_full_and_cropped_predict_file_score_"   #单个文件合并

'''softmax分数预测'''
#全图预测
# MIN_PREDICT_SCORE = 0.005  #不超出数量限制的最低分数
# EXCEED_MIN_SCORE = 0.008   #超出数量限制依然有效的分数
# SCORE_PREDICT_FILE_PREFIX = "score_full_predict_result_"

#切图预测
# MIN_PREDICT_SCORE = 0.0075  #不超出数量限制的最低分数
# EXCEED_MIN_SCORE = 0.009   #超出数量限制依然有效的分数
# SCORE_PREDICT_FILE_PREFIX = "score_merged_crop4_predict_result_"

#全图和切图综合预测
# MIN_PREDICT_SCORE = 0.005  #不超出数量限制的最低分数
# EXCEED_MIN_SCORE = 0.009   #超出数量限制依然有效的分数
# SCORE_PREDICT_FILE_PREFIX = "merged_full_and_cropped_predict_file_score_"   #多个文件合并

#单个文件预测
# MIN_PREDICT_SCORE = 0.0076  #不超出数量限制的最低分数
# EXCEED_MIN_SCORE = 0.009   #超出数量限制依然有效的分数
# SCORE_PREDICT_FILE_PREFIX = "merged_high_score_full_and_cropped_score.txt"   #单个文件合并

'''修正分数后的预测'''
#全图预测
# MIN_PREDICT_SCORE = 0.5  #不超出数量限制的最低分数
# EXCEED_MIN_SCORE = 0.6   #超出数量限制依然有效的分数
# SCORE_PREDICT_FILE_PREFIX = "score_full_predict_result_"

FINAL_PREDICT_FILE_PREFIX = "final_"
MIN_LABEL_COUNT = 2    #default=2
MAX_LABEL_COUNT = 6    #default=6
'''
class ImageLabel(object):
    def __init__(self, image, label_scores, type):
        self.image = image
        self.type = type
        label_score_array = []
        print("label_scores:",label_scores)
        for label_score in label_scores:
            score = float(label_score[label_score.find("(")+1:label_score.find(")")])
            label_score_array.append(LabelScore(label_score[:label_score.find("(")], score if type == 'full' else score/4))
        self.label_scores = label_score_array
class LabelScore(object):
    def __init__(self, label, score):
        self.label = label
        self.score = score
    def __repr__(self):
        return self.label+"("+str(self.score)+")"
    def __str__(self):
        return self.label+"("+str(self.score)+")"
def get_predict_labels(predict_file, type):
    
    predict_labels = []
    with open(predict_file) as f:
        while True:
            line = f.readline().strip()
            if line == "": break
            line_array = line.split(" ")
            predict_labels.append(ImageLabel(line_array[0], line_array[1].split(","), type))
    return predict_labels
'''
def sort_predict_file(filename):
    index = filename.split(".")[0].split("_")[-1]
    try:
        return int(index)
    except:
        return 0
def get_predict_file(score_predict_file):
    score_predict_labels = class_image_label_score.get_predict_labels(score_predict_file)
    for image_label in score_predict_labels:
        print("image:",image_label.image)
        image_label.selected_labels = get_selected_labels(image_label.label_scores)
    with open(os.path.join(SCORE_PREDICT_FILE_DIR, FINAL_PREDICT_FILE_PREFIX+score_predict_file.split("/")[-1]), "w") as f:
        for full_image_label in score_predict_labels:
            f.write(full_image_label.image+" "+re.sub("['\\[\\]\\s]","",str(full_image_label.selected_labels))+"\n")
def get_selected_labels(label_scores):
    # print("label_scores:",label_scores)
    merge_label_score_util.remove_conflict_labels(label_scores)
    selected_labels = []
    for label_score in label_scores:
        if len(selected_labels) < MIN_LABEL_COUNT:
            selected_labels.append(label_score.label)
        elif len(selected_labels) >= MAX_LABEL_COUNT:
            if label_score.score >= EXCEED_MIN_SCORE:
                selected_labels.append(label_score.label)
            continue
        else:
            if label_score.score >= MIN_PREDICT_SCORE:
                selected_labels.append(label_score.label)
    '''
    #全图预测的切图预测的前两位合并
    if len(selected_labels) < MAX_LABEL_COUNT and full_label_scores[0].label not in selected_labels and full_label_scores[0].score >= FIRST2_MULTI*MIN_PREDICT_SCORE:
        selected_labels.append(full_label_scores[0].label)
    if len(selected_labels) < MAX_LABEL_COUNT and len(full_label_scores) > 1 and full_label_scores[1].label not in selected_labels and full_label_scores[1].score >= FIRST2_MULTI*MIN_PREDICT_SCORE:
        selected_labels.append(full_label_scores[1].label)
    if len(selected_labels) < MAX_LABEL_COUNT and cropped_label_scores[0].label not in selected_labels and cropped_label_scores[0].score >= FIRST2_MULTI*MIN_PREDICT_SCORE:
        selected_labels.append(cropped_label_scores[0].label)
    if len(selected_labels) < MAX_LABEL_COUNT and cropped_label_scores[1].label not in selected_labels and cropped_label_scores[1].score >= FIRST2_MULTI*MIN_PREDICT_SCORE:
        selected_labels.append(cropped_label_scores[1].label)
    print("selected_labels:",selected_labels)
    #合并相同的标签
    for full_label_score in full_label_scores:
        for crop_label_score in cropped_label_scores:
            if full_label_score.label == crop_label_score.label and full_label_score.label not in selected_labels:
                selected_labels.append(full_label_score.label)
    #根据分数添加标签
    rest_label_scores = []
    for full_label_score in full_label_scores:
        if full_label_score.label not in selected_labels:
            rest_label_scores.append(full_label_score)
    for crop_label_score in cropped_label_scores:
        if crop_label_score.label not in selected_labels:
            rest_label_scores.append(LabelScore(crop_label_score.label, crop_label_score.score))

    rest_label_scores = sorted(rest_label_scores, key=sort_by_score, reverse=True)
    print("rest_label_scores:",rest_label_scores)
    for rest_label_score in rest_label_scores:
        if len(selected_labels) >= MAX_LABEL_COUNT or rest_label_score.score < MIN_PREDICT_SCORE:
            break
        else:
            selected_labels.append(rest_label_score.label)
    '''
    return selected_labels
def sort_by_score(label_score):
    return label_score.score
if __name__ == '__main__':
    all_files = os.listdir(SCORE_PREDICT_FILE_DIR)
    score_predict_files = []
    for filename in all_files:
        if filename.startswith(SCORE_PREDICT_FILE_PREFIX):
            score_predict_files.append(filename)
    score_predict_files = sorted(score_predict_files, key=sort_predict_file)
    for score_predict_file in score_predict_files:
        get_predict_file(os.path.join(SCORE_PREDICT_FILE_DIR, score_predict_file))

    # get_predict_file("/home/niusilong/svn/svn_repository_ai/competition_inception_v4/predict_3/merged_full_and_cropped_predict_file_score.txt")