import os
import re
SCORE_PREDICT_FILE_DIR = "/home/niusilong/work/AI/predict"
OUTPUT_PREDICT_FILE_DIR = "/home/niusilong/work/AI/predict"
SCORE_PREDICT_FILE_PREFIX = "score_full_predict_result_"
OUTPUT_PREDICT_FILE_PREFIX = "trim_"
def trim_score_predict_file():
    all_files = os.listdir(SCORE_PREDICT_FILE_DIR)
    score_predict_files = []
    for filename in all_files:
        if filename.startswith(SCORE_PREDICT_FILE_PREFIX):
            score_predict_files.append(filename)
    print("score_predict_files:",score_predict_files)
    for filename in score_predict_files:
        print(filename)
        with open(os.path.join(SCORE_PREDICT_FILE_DIR, filename)) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = re.sub("(\\(\\d{1}\\.{1}\\d+\\))","",lines[i])
                # lines[i] = re.sub("(\\(\\S\\))","",lines[i])
        with open(os.path.join(OUTPUT_PREDICT_FILE_DIR, OUTPUT_PREDICT_FILE_PREFIX+filename), 'w') as f:
            f.writelines(lines)
if __name__ == '__main__':
    trim_score_predict_file()