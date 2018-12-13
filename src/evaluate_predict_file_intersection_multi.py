import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# STANDARD_IMAGE_LABELS_PATH = "/home/niusilong/work/AI/predict/standard_competition_test_2.txt"
# STANDARD_IMAGE_LABELS_PATH = "/home/niusilong/work/AI/predict/standard_competition_120_3_besttone.txt"
# STANDARD_IMAGE_LABELS_PATH = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/standard_120_4_niusilong.txt"
STANDARD_IMAGE_LABELS_PATH = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/4_tag.txt"
TEST_IMAGE_LABELS_DIR = "/home/niusilong/work/AI/predict/high_score"
# TEST_IMAGE_LABELS_DIR = "/home/niusilong/svn/svn_repository_ai/test/competition_inception_v4/predict"
FILE_PREFIX = "final_merged_full_and_cropped_predict_file_score_"   #全图和切图统合预测文件
# FILE_PREFIX = "lat_"    #全图预测文件
# FILE_PREFIX = "predict_result_"    #全图预测文件

# FILE_PREFIX = "final_score_merged_score_full_predict_result_"   #切图预测文件
# FILE_PREFIX = "final_merged_high_score_full_and_cropped_score_"    #单个预测文件(合并全图预测和切图预测)
# FILE_PREFIX = "final_score_full_predict_result_33.txt"    #单个预测文件(全图预测)
# TEST_IMAGE_LABELS_PATH = "/home/niusilong/svn/svn_repository_ai/competition_inception_v4/predict/lat_0913_1.txt"
NUM_CLASSES = 260
def sort_img(filename):
    try:
        img_val = int(filename.split("/")[-1].split(".")[0])
        return img_val
    except:
        return filename.split("/")[-1].split(".")[0]

def evaluate_accuracy(file):
    all_standard_labels = []
    all_test_labels = []
    standard_dict = {}
    test_dict = {}
    with open(STANDARD_IMAGE_LABELS_PATH) as f:
        while True:
            line = f.readline().strip()
            if line == "":
                break
            line_array = line.split(" ")
            img = line_array[0]
            labels = line_array[1].split(",")
            standard_dict[img] = labels
    all_standard_keys = list(standard_dict.keys())
    all_standard_keys = sorted(all_standard_keys, key=sort_img, reverse=False)
    with open(file) as f:
        while True:
            line = f.readline().strip()
            if line == "":
                break
            line_array = line.split(" ")
            img = line_array[0]
            labels = [] if len(line_array) == 1 else line_array[1].split(",")
            test_dict[img] = labels

    for key in all_standard_keys:
        all_standard_labels.append(standard_dict[key])
        all_test_labels.append(test_dict[key])
    all_accuracy = []
    for i in range(len(all_standard_labels)):
        standart_labels = all_standard_labels[i]
        test_labels = all_test_labels[i]
        intersection_array = []
        sum_array = []
        for label in test_labels:
            if label in standart_labels:
                intersection_array.append(label)
            if label not in sum_array:
                sum_array.append(label)
        for label in standart_labels:
            if label not in sum_array:
                sum_array.append(label)
        accuracy = round(len(intersection_array)/len(sum_array), 5)
        all_accuracy.append(accuracy)
    # print("all_accuracy:",all_accuracy)
    final_accuracy = np.mean(all_accuracy, axis=0)
    print("final_accuracy:",round(final_accuracy, 4), file.split("/")[-1])
    return final_accuracy
def sort_by_time(filepath):
    return os.path.getmtime(filepath)
def main():
    files = os.listdir(TEST_IMAGE_LABELS_DIR)
    evaluate_files = []
    for file in files:
        if file.startswith(FILE_PREFIX):
            evaluate_files.append(os.path.join(TEST_IMAGE_LABELS_DIR, file))
    evaluate_files = sorted(evaluate_files, key=sort_by_time)
    # print(evaluate_files[0:EVALUATE_FILE_LATEST_COUNT])
    all_accuracy = []
    for i in range(len(evaluate_files)):
        accuracy = evaluate_accuracy(evaluate_files[i])
        all_accuracy.append(accuracy)
    # print("all_accuracy:",all_accuracy)
    print("np.max(all_accuracy):",round(np.max(all_accuracy), 4))
    print("mean accuracy:", round(np.mean(all_accuracy, axis=0), 4))
if __name__ == '__main__':
    main()