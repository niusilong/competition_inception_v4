import os
from file_process import class_image_label_score
PREDICT_DIR = "/home/niusilong/work/AI/predict"
PREDICT_FILE_PREFIX = "score_merged_score_full_predict_result_"

all_files = os.listdir(PREDICT_DIR)
predict_files = []
for filename in all_files:
    if filename.startswith(PREDICT_FILE_PREFIX):
        predict_files.append(filename)
for filename in predict_files:
    predict_infos = class_image_label_score.get_predict_labels(os.path.join(PREDICT_DIR, filename))