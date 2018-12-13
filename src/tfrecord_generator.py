#coding=utf-8
'''
多线程导出tfrecord文件
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os.path
import threading
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# tf.app.flags.DEFINE_string('label_file_path', "/home/niusilong/work/AI/training/image_labels_niusilong.txt", """""")
# tf.app.flags.DEFINE_string('label_file_path', "/home/niusilong/projects/image-cut/src/main/webapp/training_cropped/training_cropped_labels_setted.txt", """""")
# tf.app.flags.DEFINE_string('label_file_path', "/home/niusilong/projects/image-cut/src/main/webapp/training/training_labels_modified.txt", """""")
# tf.app.flags.DEFINE_string('label_file_path', "/home/niusilong/projects/image-cut/src/main/webapp/training/training_labels.txt", """""")
# tf.app.flags.DEFINE_string('label_file_path', "/home/niusilong/work/AI/label_files/appointed_images_3.txt", """""")
# tf.app.flags.DEFINE_string('label_file_path', "/home/niusilong/svn/svn_repository_ai/competition_inception_v4/label_file/appointed_test_2_2.txt", """""")
# tf.app.flags.DEFINE_string('label_file_path', "/home/niusilong/projects/image-cut/src/main/resources/label_file/crop4_120_2_128.txt", """""")
tf.app.flags.DEFINE_string('label_file_path', "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/resources/label_file/crop4_120_3_128.txt", """""")
tf.app.flags.DEFINE_string('tfrecord_dir', "/home/niusilong/work/AI/tfrecords", """""")
tf.app.flags.DEFINE_string('tfrecord_name', "crop4_120_3_128", """""")
tf.app.flags.DEFINE_integer('image_size', 128, """""")
tf.app.flags.DEFINE_integer('num_classes', 260, """""")
tf.app.flags.DEFINE_integer('total_parts', 1, """为了防止文件过大导致内在不足，分开存储多个tfrecord文件""")
tf.app.flags.DEFINE_integer('image_start_index', 0, """图片开始位置""")
tf.app.flags.DEFINE_integer('image_end_index', 100000, """图片结束位置""")
FLAGS = tf.app.flags.FLAGS
def _string_feature(value):
    return tf.train.Feature()
class BaseImageData(object):
    def __init__(self, labels, image_path):
        self.labels = labels
        self.image_path = image_path
def action(tfrecord_path, sub_image_array):
    with tf.Session() as sess:
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for i in range(len(sub_image_array)):
            image_path = sub_image_array[i].image_path
            file = tf.gfile.FastGFile(image_path, 'rb')
            # print("i:",i,"image_path: ", image_path)
            image_raw_data = file.read()
            image = tf.image.decode_jpeg(image_raw_data)
            # print("original shape: ", tf.shape(image).eval())
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, size=[FLAGS.image_size, FLAGS.image_size], method=0)
            last_image_data = image
            # print("shape: ", shape)
            # print("image: ", image.eval())
            labels = sub_image_array[i].labels
            multi_hot_label = get_multi_hot(labels)
            # print("image.eval().tostring(): ", image.eval().tostring())
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.eval().tostring()])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(labels, encoding='utf-8')])),
                    'multi_hot_label': tf.train.Feature(float_list=tf.train.FloatList(value=multi_hot_label)),
                    'image_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(image_path, encoding='utf-8')]))
                }
            ))
            writer.write(example.SerializeToString())
            if i % 10 == 0: print("%d/%d writed, %s." % (i, len(sub_image_array), tfrecord_path))
        writer.close()
def get_multi_hot(labels):
    multi_hot = np.zeros(FLAGS.num_classes, dtype=np.float32)
    if labels == "":
        return multi_hot
    label_array = [int(x)-1 for x in labels.split(",")]
    for label in label_array:
        multi_hot[label] = 1.0
    return multi_hot
def main():
    image_data_array = []
    with open(FLAGS.label_file_path) as f:
        while True:
            line = f.readline().strip()
            if line == '':
                break
            # print(line)
            line_array = line.split(" ")
            image_data = BaseImageData(labels=(line_array[1] if len(line_array) == 2 else ""), image_path=line_array[0])
            image_data_array.append(image_data)
    print("图片总数：",len(image_data_array))
    image_data_array = image_data_array[FLAGS.image_start_index:FLAGS.image_end_index]
    print("len(image_data_array):",len(image_data_array))
    image_count_per_part = int(len(image_data_array)/FLAGS.total_parts)
    for i in range(FLAGS.total_parts):
        start_index = i*image_count_per_part
        end_index = (i+1)*image_count_per_part
        if (i+2)*image_count_per_part > len(image_data_array):
            end_index = len(image_data_array)
        tfrecord_path = os.path.join(FLAGS.tfrecord_dir, FLAGS.tfrecord_name+"_part_"+str(i)+".tfrecords")
        t =threading.Thread(target=action,args=(tfrecord_path, image_data_array[start_index:end_index],))
        t.start()
        if end_index >= len(image_data_array):
            break
if __name__ == '__main__':
    # with tf.device('/cpu:0'):
        main()
