appointed_labels = ['233', '75', '203', '245', '225', '32', '186', '22', '79', '197', '107', '134', '244', '90', '240', '26', '167', '150', '242', '236', '136', '164', '53', '188', '132', '241', '43', '122', '214', '139', '158', '207', '170', '28', '199', '124', '260', '84', '54', '14', '165', '127', '249', '205', '168', '71', '228', '105', '206', '93', '99', '98', '85', '194', '59', '180', '45', '226', '195', '163', '17', '209', '184', '204', '259', '29', '175', '12', '67', '210', '58', '80', '25', '181', '196', '81', '19', '72']
SRC_LABEL_FILE = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/4740_absolute_path_setted.txt"
write_file_path = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/appointed_120_4.txt"
selected_labels = []
appointed_image_labels = []
def get_selected_count(selected_labels, appoint_label):
    count = 0
    for label in selected_labels:
        if appoint_label == label:
            count += 1
    return count

with open(SRC_LABEL_FILE) as f:
    image_label_lines = f.readlines()
    image_label_lines = [line.strip() for line in image_label_lines]
for appoint_label in appointed_labels:
    line_index = 0
    while True:
        if get_selected_count(selected_labels, appoint_label) >= 30:
            break
        line = image_label_lines[line_index]
        line_array = line.split(" ")
        if len(line_array) == 1:
            continue
        label_array = line_array[1].split(",")
        if appoint_label in label_array:
            appointed_image_labels.append(line)
            selected_labels.extend(label_array)
        line_index += 1
        if line_index >= len(image_label_lines):
            line_index = 0
# with open(SRC_LABEL_FILE) as f:
#     while True:
#         line = f.readline().strip()
#         if line == "":
#             break
#         line_array = line.split(" ")
#         if len(line_array) == 1:
#             continue
#         label_array = line_array[1].split(",")
#         for label in label_array:
#             if label in appointed_labels and get_selected_count(selected_labels, label) < 15:
#                 appointed_image_labels.append(line)
#                 selected_labels.extend(label_array)
#                 break
# selected_labels = list(set(selected_labels))
# print("selected_labels:",selected_labels)
for label in appointed_labels:
    print(label, get_selected_count(selected_labels, label))
with open(write_file_path, 'w') as f:
    for line in appointed_image_labels:
        f.write(line+"\n")