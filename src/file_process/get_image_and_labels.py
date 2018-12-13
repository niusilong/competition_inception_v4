import os
IMG_DIR = "/home/niusilong/work/AI/dataset_1-6"

def sort_image(filename):
    try:
        return int(filename[filename.find("_")+1:filename.find(".")])
    except:
        return 0
def get_image_label(dir):
    all_files = os.listdir(dir)
    dirname = dir.split("/")[-1]
    all_files = sorted(all_files, key=sort_image)
    for filename in all_files:
        print(os.path.join(dir, filename)+" "+dirname.split("-")[1])
def get_image_labe_parent(parent_dir):
    dirs = os.listdir(parent_dir)
    for dir in dirs:
        get_image_label(os.path.join(IMG_DIR, dir))
if __name__ == '__main__':
    get_image_labe_parent(IMG_DIR)