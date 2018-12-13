import os
def sort_by_time(filepath):
    return os.path.getmtime(filepath)
def sort_by_file_index(filename):
    index = filename.split(".")[0].split("_")[-1]
    try:
        return int(index)
    except:
        return 0