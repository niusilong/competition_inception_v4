from file_process import get_label_count
APPOINTED_LABEL_FILE = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/3_tag.txt"
ALL_IMAGE_FILE = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/4740_absolute_path.txt"
all_appointed_labels = []
with open(APPOINTED_LABEL_FILE) as f:
    while True:
        line = f.readline().strip()
        if line == "":break
        line_array = line.split(" ")
        if len(line_array) < 2:
            continue
        labels = line_array[1].split(",")
        labels = [int(label) for label in labels]
        for label in labels:
            if label not in all_appointed_labels:
                all_appointed_labels.append(label)
print("len(all_appointed_labels):",len(all_appointed_labels))
print(all_appointed_labels)

with open(ALL_IMAGE_FILE) as f:
    all_image_lines = f.readlines()
all_image_label_count = get_label_count.get_label_count(all_image_lines)
# print(all_image_label_count)
def sort_label(label_id):
    return all_image_label_count[label_id-1]
all_appointed_labels = sorted(all_appointed_labels, key=sort_label)
print("all_appointed_labels:",all_appointed_labels)