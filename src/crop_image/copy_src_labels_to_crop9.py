SRC_LABEL="/home/niusilong/projects/image-cut/src/main/resources/label_file/3_tag_absolute_path.txt"
DEST_LABEL_PATH = "/home/niusilong/projects/image-cut/src/main/resources/label_file/3_tag_crop9_absolute_path.txt"
with open(SRC_LABEL) as f:
    lines = f.readlines()
with open(DEST_LABEL_PATH, "w") as f:
    for line in lines:
        dot_index = line.find(".")
        for i in range(9):
            f.write(line[:dot_index]+"_"+str(i)+line[dot_index:])