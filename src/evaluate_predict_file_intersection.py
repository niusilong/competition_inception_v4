import numpy as np
STANDARD_IMAGE_LABELS_PATH = "/home/niusilong/work/AI/predict/standard_competition_120_3_besttone.txt"
# TEST_IMAGE_LABELS_PATH = "/home/niusilong/svn/svn_repository_ai/competition_inception_v4/predict_3/merged_full_and_cropped_predict_file.txt"
# TEST_IMAGE_LABELS_PATH = "/home/niusilong/svn/svn_repository_ai/competition_inception_v4/predict_3/final_merged_full_and_cropped_predict_file_score.txt"
# TEST_IMAGE_LABELS_PATH = "/home/niusilong/work/AI/predict/final_merged_all_score_full_predict_result.txt"
# TEST_IMAGE_LABELS_PATH="/home/niusilong/svn/svn_repository_ai/competition_inception_v4/predict_3/lat_1011_3.txt"
# TEST_IMAGE_LABELS_PATH = "/home/niusilong/svn/svn_repository_ai/competition_inception_v4/predict/lat_0927_3.txt"
# TEST_IMAGE_LABELS_PATH = "/home/niusilong/work/AI/predict/final_merged_all_score_full_predict_result.txt"
TEST_IMAGE_LABELS_PATH = "/home/niusilong/work/AI/predict/final_merged_high_score_full_and_cropped_score.txt"
# TEST_IMAGE_LABELS_PATH = "/home/niusilong/ftp/result_ludong.txt"

NUM_CLASSES = 260
def sort_img(filename):
    try:
        img_val = int(filename.split("/")[-1].split(".")[0])
        return img_val
    except:
        return filename.split("/")[-1].split(".")[0]

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
with open(TEST_IMAGE_LABELS_PATH) as f:
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
    print("i:%d, key:%s, standard_labels:%s, test_labels:%s, intersection_labels:%s, sum_labels:%s, accuracy:%.4f" % (i, all_standard_keys[i], all_standard_labels[i], all_test_labels[i], intersection_array, sum_array, accuracy))
    all_accuracy.append(accuracy)
print("all_accuracy:",all_accuracy)
all_keys = list(all_standard_keys)
for i in range(len(all_keys)):
    print("key:%s, accuracy:%.4f" % (all_keys[i], all_accuracy[i]))
final_accuracy = np.mean(all_accuracy, axis=0)
print("np.sum(all_accuracy):",round(np.sum(all_accuracy), 4))
print("final_accuracy:",round(final_accuracy, 4), TEST_IMAGE_LABELS_PATH.split("/")[-1])
