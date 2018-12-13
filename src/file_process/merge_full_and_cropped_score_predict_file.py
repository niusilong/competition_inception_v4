'''
合并不带分数的全图预测和切图预测
'''
from file_process import class_image_label_score
from file_process.strategy import merge_label_score_util
import re
FULL_PREDICT_FILE = "/home/niusilong/work/AI/predict/high_score/score_full_predict_result_33.txt"
CROPPED_PREDICT_FILE = "/home/niusilong/work/AI/predict/high_score/score_merged_score_full_predict_result_19.txt"
OUTPUT_PREDICT_FILE = "/home/niusilong/work/AI/predict/merged_high_score_full_and_cropped_score.txt"

def get_predict_labels(predict_file):
    predict_labels = []
    with open(predict_file) as f:
        while True:
            line = f.readline().strip()
            if line == "": break
            line_array = line.split(" ")
            predict_labels.append(class_image_label_score.ImageLabel(line_array[0], line_array[1].split(",")))
    return predict_labels
def merge_predict_file(full_predict_file, cropped_predict_file, output_filename):
    full_predict_info = get_predict_labels(full_predict_file)
    cropped_predict_info = get_predict_labels(cropped_predict_file)
    merged_predict_info = []
    for merged_image_label in full_predict_info:
        for cropped_image_label in cropped_predict_info:
            if merged_image_label.image == cropped_image_label.image:
                # merge_label_score_util.remove_impossible_labels(merged_image_label.label_scores, cropped_image_label.label_scores)
                # merge_label_score_util.reset_score_proportion(cropped_image_label.label_scores, 0.9)
                # merged_label_scores = merge_label_score_util.merge_label_score_by_max_score(merged_image_label.label_scores, cropped_image_label.label_scores)
                # merged_label_scores = merge_label_score_util.merge_label_score_by_proportion(merged_image_label.label_scores, cropped_image_label.label_scores, full_score_proportion=0.5, crop_score_propertion=0.5)
                merged_label_scores = merge_label_score_util.merge_label_score_by_max_score_with_first_two_and_same_label(merged_image_label.label_scores, cropped_image_label.label_scores, default_first_two_score_full=0.9, default_first_two_score_crop=0.7)
                # merged_label_scores = merge_label_score_util.merge_label_score_by_min_score(merged_image_label.label_scores, cropped_image_label.label_scores, full_min_score=[1.0, 0.8], crop_min_score=[1.0, 0.8])
                merged_predict_info.append(class_image_label_score.ImageLabel(merged_image_label.image, label_scores=merged_label_scores))
    with open(output_filename, "w") as f:
        for merged_image_label in merged_predict_info:
            f.write(merged_image_label.image+" "+re.sub("['\\[\\]\\s]","",str(merged_image_label.label_scores))+"\n")
if __name__ == '__main__':
    merge_predict_file(FULL_PREDICT_FILE, CROPPED_PREDICT_FILE, OUTPUT_PREDICT_FILE)