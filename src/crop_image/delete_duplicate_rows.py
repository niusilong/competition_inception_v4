FILE_PATH = "/home/niusilong/projects/image-cut/src/main/resources/label_file/1_tag_crop9_absolute_path_setted.txt"
with open(FILE_PATH) as f:
    lines = f.readlines()
print("len(lines):",len(lines))
filepath_array = []
for line in lines:
    if line == "": break
    filepath_array.append(line.split(" ")[0])
print("len(filepath_array):",len(filepath_array))
to_delete_index = []
for i in range(len(filepath_array)-1):
    if i in to_delete_index:
        continue
    for j in range(i+1, len(filepath_array)):
        if filepath_array[i] == filepath_array[j]:
            # print("i:%d, j:%d" % (i, j))
            to_delete_index.append(j)
to_delete_index.sort(reverse=True)
print(to_delete_index)
for index in to_delete_index:
    lines.pop(index)
print("len(lines):",len(lines))
with open(FILE_PATH, 'w') as f:
    f.writelines(lines)