import numpy as np
def get_all_sorted_array(array):
    if len(array) == 1:
        return [array]
    all_sorted_array = []
    for i in range(len(array)):
        rest_array = []
        for j in range(len(array)):
            if j == i:
                continue
            else:
                rest_array.append(array[j])
        all_child_sorted_array = get_all_sorted_array(rest_array)
        for child_sorted_array in all_child_sorted_array:
            sorted_array = [array[i]]
            sorted_array.extend(child_sorted_array)
            all_sorted_array.append(sorted_array)
    return all_sorted_array
def get_all_sorted_array_flattern(array):
    return np.ravel(get_all_sorted_array(array))
if __name__ == '__main__':
    array = [1,2,3,4]
    all_sorted_array = get_all_sorted_array(array)
    print(all_sorted_array)
    print(get_all_sorted_array_flattern(array))