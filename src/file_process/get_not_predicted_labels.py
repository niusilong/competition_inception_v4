import re
STANDARD_LABEL_FILE="/home/niusilong/work/AI/predict/standard_competition_120_2_besttone.txt"
# PREDICT_LABEL_FILE="/home/niusilong/svn/svn_repository_ai/competition_inception_v4/predict_2/lat_0927_2.txt"
PREDICT_LABEL_FILE="/home/niusilong/work/AI/predict/predict_result_259.txt"
standard_labels_dict={}
predict_labels_dict={}
with open(STANDARD_LABEL_FILE) as f:
    while True:
        line = f.readline().strip()
        if line == "":break
        line_array = line.split(" ")
        standard_labels_dict[line_array[0]] = line_array[1].split(",")
with open(PREDICT_LABEL_FILE) as f:
    while True:
        line = f.readline().strip()
        if line == "":break
        line_array = line.split(" ")
        predict_labels_dict[line_array[0]] = line_array[1].split(",")
images = standard_labels_dict.keys()
not_predicted_labels = []
for image in images:
    standard_labels =  standard_labels_dict[image]
    predict_labels = predict_labels_dict[image]
    for label in standard_labels:
        if label == '0': print(image)
        if label not in predict_labels and label not in not_predicted_labels:
                not_predicted_labels.append(label)
print(re.sub('[\']','',str(not_predicted_labels)))
# print(not_predicted_labels)