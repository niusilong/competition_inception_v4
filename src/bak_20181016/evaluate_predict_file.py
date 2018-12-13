import numpy as np
import tensorflow as tf

STANDARD_IMAGE_LABELS_PATH = "/home/niusilong/work/AI/predict/standard_image_labels.txt"
TEST_IMAGE_LABELS_PATH = "/home/niusilong/work/AI/predict/predict_result_83.txt"
NUM_CLASSES = 260
def get_multi_hot(labels_array):
    multi_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
    if labels == None or len(labels_array) == 0:
        return multi_hot
    # label_array = [int(x)-1 for x in labels.split(",")]
    for label in labels_array:
        multi_hot[int(label)-1] = 1.0
    return multi_hot
standard_multi_hot = []
test_multi_hot = []
standard_dict = {}
test_dict = {}
with open(STANDARD_IMAGE_LABELS_PATH) as f:
    while True:
        line = f.readline().strip()
        if line == "":
            break
        line_array = line.split(" ")
        img = line_array[0]
        labels = line_array[1].split(",")
        standard_dict[img] = labels
with open(TEST_IMAGE_LABELS_PATH) as f:
    while True:
        line = f.readline().strip()
        if line == "":
            break
        line_array = line.split(" ")
        img = line_array[0]
        labels = [] if len(line_array) == 1 else line_array[1].split(",")
        test_dict[img] = labels

for key in standard_dict.keys():
    standard_multi_hot.append(get_multi_hot(standard_dict[key]))
    test_multi_hot.append(get_multi_hot(test_dict[key]))
standard_multi_hot = np.asarray(standard_multi_hot)
test_multi_hot = np.asarray(test_multi_hot)
print("shape:",np.shape(standard_multi_hot), np.shape(test_multi_hot))
correct_prediction = tf.equal(tf.constant(test_multi_hot, shape=[len(standard_multi_hot), NUM_CLASSES], dtype=tf.float32), tf.constant(standard_multi_hot, shape=[len(standard_multi_hot), NUM_CLASSES], dtype=tf.float32))
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
with tf.Session() as sess:
    print(sess.run(evaluation_step))