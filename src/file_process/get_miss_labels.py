from file_process import class_image_label_score
from file_process import get_label_count
ALL_IMAGE_FILE = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/4860_absolute_path_setted.txt"
STANDARD_LABEL_FILE = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/standard_120_4_niusilong.txt"
PREDICT_LABEL_FILE = "/home/niusilong/work/AI/predict/high_score/final_score_full_predict_result_33.txt"

standard_predict_info = class_image_label_score.get_final_predict_labels(STANDARD_LABEL_FILE)
predict_predict_info = class_image_label_score.get_final_predict_labels(PREDICT_LABEL_FILE)

def get_miss_labels(standard_labels, predict_labels):
    miss_labels = []
    for standard_label in standard_labels:
        if standard_label not in predict_labels:
            miss_labels.append(standard_label)
    return miss_labels
with open(ALL_IMAGE_FILE) as f:
    all_image_lines = f.readlines()
all_image_label_count = get_label_count.get_label_count(all_image_lines)
# print(all_image_label_count)
def sort_label(label_id):
    return all_image_label_count[int(label_id)-1]
all_miss_labels = []
for standard_image_labels in standard_predict_info:
    for predict_image_labels in predict_predict_info:
        if standard_image_labels.image == predict_image_labels.image:
            miss_labels = get_miss_labels(standard_image_labels.labels, predict_image_labels.labels)
            all_miss_labels.extend(miss_labels)
print(len(all_miss_labels))
print(all_miss_labels)
all_miss_labels = list(set(all_miss_labels))
print(len(all_miss_labels))
print(all_miss_labels)
all_miss_labels = sorted(all_miss_labels, key=sort_label)
print("-------------------------------")
print(all_miss_labels)
print("-------------------------------")
