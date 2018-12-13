'''
合并不带分数的全图预测和切图预测
'''
from file_process import merge_full_and_cropped_score_predict_file
from file_process.strategy import file_sort_util
from file_process.strategy import get_all_sorted_array
import os
import random
SCORE_PREDICT_FILE_DIR = '/home/niusilong/work/AI/predict/high_score'
FULL_PREDICT_FILE_PREFIX = "score_full_predict_result_"
CROP_PREDICT_FILE_PREFIX = "score_merged_score_full_predict_result_"

OUTPUT_PREDICT_FILE_NAME = "merged_full_and_cropped_predict_file_score"

if __name__ == '__main__':
    all_files = os.listdir(SCORE_PREDICT_FILE_DIR)
    full_score_predict_files = []
    crop_score_predict_files = []
    for filename in all_files:
        if filename.startswith(FULL_PREDICT_FILE_PREFIX):
            full_score_predict_files.append(filename)
        if filename.startswith(CROP_PREDICT_FILE_PREFIX):
            crop_score_predict_files.append(filename)
    full_score_predict_files = sorted(full_score_predict_files, key=file_sort_util.sort_by_file_index)
    crop_score_predict_files = sorted(crop_score_predict_files, key=file_sort_util.sort_by_file_index)
    print("full_score_predict_files:",full_score_predict_files)
    if len(full_score_predict_files) != len(crop_score_predict_files):
        raise Exception("全力预测文件和切图预测文件数量不相同"+str(len(full_score_predict_files))+", "+str(len(crop_score_predict_files)))
    all_sorted_crop_files = get_all_sorted_array.get_all_sorted_array_flattern(crop_score_predict_files)

    extended_full_files = []
    while True:
        if len(extended_full_files) < len(all_sorted_crop_files):
            extended_full_files.extend(full_score_predict_files)
        else:
            break
    print("len(extended_full_files)",len(extended_full_files), "len(all_sorted_crop_files):", len(all_sorted_crop_files))
    for i in range(len(extended_full_files)):
        merge_full_and_cropped_score_predict_file.merge_predict_file(
            os.path.join(SCORE_PREDICT_FILE_DIR, extended_full_files[i]),
            os.path.join(SCORE_PREDICT_FILE_DIR, all_sorted_crop_files[i]),
            os.path.join(SCORE_PREDICT_FILE_DIR, OUTPUT_PREDICT_FILE_NAME+"_"+str(i+1)+".txt")
        )