import os
import re
NINE_LABEL_FILE = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/3_tag_crop9_absolute_path_setted.txt"
FOUR_LABEL_FILE = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/crop4_120_3_128.txt"
IMAGE_FILE_PREFIX_9 = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/webapp/images/crop9_120_3/"
IMAGE_FILE_PREFIX_4 = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/webapp/images/crop4_120_3_128/"
PARENT_IMAGE_DIR = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/webapp/images/120_3"
image_label_array_9 = []
all_parent_image = []
class ImageLabel(object):
    def __init__(self, image, labels):
        self.image = image
        self.labels = labels
with open(NINE_LABEL_FILE) as f:
    while True:
        line = f.readline().strip()
        if line == "":
            break
        line_array = line[len(IMAGE_FILE_PREFIX_9):].split(" ")
        image = line_array[0].split(".")[0]
        image_label = ImageLabel(image, line_array[1].split(",") if len(line_array) == 2 else [])
        parent_image = int(image[:image.find("_")])
        if parent_image not in all_parent_image:
            all_parent_image.append(parent_image)
        image_label_array_9.append(image_label)
print("len(image_label_array_9):",len(image_label_array_9))
parent_images_list = os.listdir(PARENT_IMAGE_DIR)
original_parent_array = []
for image in parent_images_list:
    original_parent_array.append(int(image[:image.find(".")]))
original_parent_array.sort()
image_label_array_4 = []
image_label_4_lines = []

index_map = [[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]]
def get_image_labels_line(image, index):
    sum_lables_9 = []
    processed_count = 0
    index_9_array = index_map[index]
    for image_label in image_label_array_9:
        for index_9 in index_9_array:
            if image_label.image == image+"_"+str(index_9):
                # print("image_label.image:",image_label.image)
                labels_9 = image_label.labels
                for label in labels_9:
                    if label not in sum_lables_9:
                        sum_lables_9.append(label)
                # image_label_array_9.remove(image_label)
                processed_count += 1
        if processed_count == 4:
            break
    line = IMAGE_FILE_PREFIX_4+str(image)+"_"+str(index)+".jpeg "+re.sub("[\s\[\]']", "", str(sum_lables_9))
    return line
for i in range(len(original_parent_array)):
    image = str(original_parent_array[i])
    image_label_4_lines.append(get_image_labels_line(image, 0))
    image_label_4_lines.append(get_image_labels_line(image, 1))
    image_label_4_lines.append(get_image_labels_line(image, 2))
    image_label_4_lines.append(get_image_labels_line(image, 3))
    to_delete_array = []
    for j in range(len(image_label_array_9)):
        image_label = image_label_array_9[j]
        if image_label.image.startswith(image+"_"):
            to_delete_array.append(j)
        if len(to_delete_array) >= 9:
            break
    to_delete_array.sort(reverse=True)
    for k in to_delete_array:
        image_label_array_9.pop(k)
    print("len(image_label_4_lines):",len(image_label_4_lines), "len(image_label_array_9):",len(image_label_array_9))

# for i in range(len(image_label_4_lines)):
#     print(image_label_4_lines[i])
with open(FOUR_LABEL_FILE, 'w') as f:
    for line in image_label_4_lines:
        f.write(line+"\n")