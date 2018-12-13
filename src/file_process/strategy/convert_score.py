import numpy as np
# def convert_to_standard_score(score_array):
#     first_score = score_array[0]
#     multi_para = 1 / first_score
#     for i in range(len(score_array)):
#         score_array[i] = score_array[i]*multi_para

def convert_to_standard_score_2dim(score_array):
    for i in range(len(score_array)):
        print("score_array[i]:",score_array[i])
        convert_to_standard_score(score_array[i])
        print("score_array[i]:",score_array[i])
def convert_to_standard_score(score_array, multi_data=None):
    # print("multi_data:",multi_data)
    if multi_data == None:
        multi_data = 1 / score_array[0]
    for i in range(len(score_array)):
        score_array[i] = score_array[i]*multi_data
if __name__ == '__main__':
    score = [0.5,0.4,0.3,0.2,0.1]
    print(score)
    score2 = [[0.5,0.4,0.3,0.2,0.1], [0.5,0.4,0.3,0.2,0.1]]
    print("score2:",score2)
    print("score2[0]:",score2[0])
    convert_to_standard_score_2dim(score2)
    print(score2)