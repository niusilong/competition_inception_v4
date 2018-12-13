#coding=utf-8
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import change_image_util
from nets import inception_v4
from tensorflow.python.framework import graph_util
#加载通过Tensorflow-Slim定义好的inception_v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

import matplotlib.pyplot as plt
#处理好之后的文件
#保存训练好之后的路径，这里可以将使用新数据训练得到的完整模型保存下来，如果计算资源充足，还可以在训练完最后的全连接层之后再训练所有网络层，这样可以使得新模型更加贴近数据

FLAGS = tf.app.flags.FLAGS
#自定义参数，可以在sh脚本文件中修改
tf.app.flags.DEFINE_integer('train_steps', 1000, """""")
tf.app.flags.DEFINE_integer('image_size', 192, """""")
tf.app.flags.DEFINE_integer('batch_size', 30, """""")
tf.app.flags.DEFINE_integer('num_classes', 260, """""")
tf.app.flags.DEFINE_string('tfrecord_file_pattern', "/home/niusilong/work/AI/tfrecords/competition_part_*.tfrecords", """包含图片，标签，multi-hot数据的压缩包""")
tf.app.flags.DEFINE_string('ckpt_save_dir', "/home/niusilong/work/AI/ckpt/multi-label-competition_2", """保存训练后的模板数据目录""")
tf.app.flags.DEFINE_string('ckpt_save_file_name', "multi-label-competition.ckpt", """模板文件名""")
tf.app.flags.DEFINE_string('output_graph', '/home/niusilong/work/AI/graph/multi-label-competition_2.pb', """Where to save the trained graph.""")
tf.app.flags.DEFINE_float('learning_rate_base', 0.0001, """初始学习率""")
tf.app.flags.DEFINE_float('learning_rate_decay', 0.98, """学习率的衰减率""")
tf.app.flags.DEFINE_integer('learning_rate_decay_steps', 100, """衰减步长""")
tf.app.flags.DEFINE_integer('random_brightness', 0, """A percentage determining how much to randomly multiply the training image input pixels up or down by.""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result', """The name of the output classification layer in the retrained graph.""")
tf.app.flags.DEFINE_string('image_batch_name', 'image_batch', """The name of the output classification layer in the retrained graph.""")
tf.app.flags.DEFINE_string('accuracy_name', 'evaluation/accuracy', """The name of the output classification layer in the retrained graph.""")

CHECKPOINT_EXCLUDE_SCOPES = "InceptionV3/Logits,InceptionV3/AuxLogits"
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
#需要训练的网络层参数名称，在fine-tuning的过程中就是最后的全连接层，这里给出的是参数的前缀
TRAINABLE_SCOPE = "InceptionV3/Logits,InceptionV3/Auxlogits"
#获取所有需要从从谷歌训练好的模型中加载的参数

def get_image_batch():
    files = tf.train.match_filenames_once(FLAGS.tfrecord_file_pattern)

    filename_queue = tf.train.string_input_producer(files, num_epochs=150, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'multi_hot_label': tf.FixedLenFeature([FLAGS.num_classes], tf.float32),
            'image_path': tf.FixedLenFeature([], tf.string)
        }
    )
    image, label, multi_hot_label = features['image'], features['label'], features['multi_hot_label']
    #从原始图像解析出像素矩阵，并根据图像尺寸还原图像
    decoded_image = tf.decode_raw(image, tf.float32)
    decoded_image = tf.reshape(decoded_image, shape=[FLAGS.image_size, FLAGS.image_size, 3])
    #定义神经网络输入层图片的大小
    distorted_image = change_image_util.preprocess_for_train(decoded_image, height=FLAGS.image_size, width=FLAGS.image_size, bbox=None, random_brightness=FLAGS.random_brightness)
    # distorted_image = decoded_image
    distorted_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])   #设置shape，否则shuffle_batch方法会报找不到shape错误
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * FLAGS.batch_size
    image_batch, label_batch, multi_hot_label_batch = tf.train.shuffle_batch([distorted_image, label, multi_hot_label], batch_size=FLAGS.batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue, name="image_batch")
    return image_batch, label_batch, multi_hot_label_batch

def main(argv=None):
    #加载预处理好的数据
    #定义inception_v3的输入,
    images, labels, multi_hot_labels = get_image_batch()
    # labels = tf.one_hot(labels, depth=5)
    #定义inception_v3模型，因为谷歌给出的只有模型参数取值，所以这里需要在这个代码中定义inception_v3的模型结构。虽然理论上需要区分训练和
    #测试中使用的模型，也就是说在测试时应该使用is_training=False, 但是因为预预先训练好的inception-v3模型中使用的batch normalization参数
    #与新的数据会有差异，导致结果很差，所以这里直接使用同一个模型来进行测试
    global_step = tf.Variable(0, trainable=False)
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits, _ = inception_v4.inception_v4(images, num_classes=FLAGS.num_classes)
    #获取需要训练的变量
    final_tensor = tf.nn.sigmoid(logits, name=FLAGS.final_tensor_name)
    tf.summary.histogram(FLAGS.final_tensor_name + '/activations', final_tensor)
    #定义交叉熵损失，注意在模型定义的时候已经将正则化损失加入损失集合了
    tf.losses.sigmoid_cross_entropy(multi_class_labels=multi_hot_labels, logits=logits, weights=1.0)

    #定义训练过程。这里minimize的过程中指定了需要优化的变量集合
    # learning_rate = LEARNING_RATE_BASE
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_base, global_step=global_step, decay_steps=FLAGS.learning_rate_decay_steps, decay_rate=FLAGS.learning_rate_decay)
    loss = tf.losses.get_total_loss()
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.round(final_tensor), multi_hot_labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


    #定义保存新的训练好的模型函数
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #初始化没有加载进来的变量。注意这个过程一定要在模型加载之前，否则初始化过程会将已经加载好的变量重新加载
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print("restored global_step:",global_step.eval())
        #保存graph
        for i in range(FLAGS.train_steps):
            #运行训练过程，这里不会更新全部的参数，只会更新指定的部分参数
            sess.run(train_step)
            if (i+1) % 100 == 0 or (i+1) == FLAGS.train_steps:
                saver.save(sess=sess, save_path=os.path.join(FLAGS.ckpt_save_dir, FLAGS.ckpt_save_file_name))
                validation_accuracy = sess.run(evaluation_step)
                print("Step %d: Validation accuracy = %.1f%%" % (global_step.eval(), validation_accuracy*100.0))
        coord.request_stop()
        coord.join(threads)
        # print("evaluation_step.name:",evaluation_step.name)
        # print("multi_hot_labels.name：",multi_hot_labels.name)
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [FLAGS.image_batch_name, FLAGS.final_tensor_name, FLAGS.accuracy_name])
        with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
if __name__ == '__main__':
    tf.app.run()