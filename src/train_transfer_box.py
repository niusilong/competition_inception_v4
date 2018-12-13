#coding=utf-8
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import change_image_util
from tensorflow.python.framework import graph_util
#加载通过Tensorflow-Slim定义好的inception_v3模型
from nets import inception_v4
import time
import matplotlib.pyplot as plt
#处理好之后的文件
#保存训练好之后的路径，这里可以将使用新数据训练得到的完整模型保存下来，如果计算资源充足，还可以在训练完最后的全连接层之后再训练所有网络层，这样可以使得新模型更加贴近数据

FLAGS = tf.app.flags.FLAGS
#自定义参数，可以在sh脚本文件中修改
tf.app.flags.DEFINE_integer('train_steps', 1000, """""")
tf.app.flags.DEFINE_integer('image_size', 192, """""")
tf.app.flags.DEFINE_integer('batch_size', 30, """""")
tf.app.flags.DEFINE_integer('num_classes', 260, """""")
#tf.app.flags.DEFINE_string('initial_ckpt_file', "D:/MyWorkspace/MyPrograms/tensorflow/models-ckpt/inception_v4.ckpt", """训练初始模板""")
tf.app.flags.DEFINE_string('initial_ckpt_file', "/home/niusilong/work/AI/ckpt/transfer_ckpt/inception_v4.ckpt", """训练初始模板""")

tf.app.flags.DEFINE_string('tfrecord_file_pattern', "/home/niusilong/work/AI/tfrecords/resetted_*.tfrecords", """包含图片，标签，multi-hot数据的压缩包""")
tf.app.flags.DEFINE_string('ckpt_save_dir', "/home/niusilong/work/AI/ckpt/competition-inception_v4_4860_resetted_appointed", """保存训练后的模板数据目录""")
tf.app.flags.DEFINE_string('ckpt_save_file_name', "competition.ckpt", """模板文件名""")
tf.app.flags.DEFINE_string('output_graph', '/home/niusilong/work/AI/graph/competition-inception_v4_4860_resetted_appointed.pb', """Where to save the trained graph.""")


# tf.app.flags.DEFINE_string('tfrecord_file_pattern', "/home/niusilong/work/AI/tfrecords/crop4_*.tfrecords", """包含图片，标签，multi-hot数据的压缩包""")
# tf.app.flags.DEFINE_string('ckpt_save_dir', "/home/niusilong/work/AI/ckpt/competition-inception_v4_4860_crop4", """保存训练后的模板数据目录""")
# tf.app.flags.DEFINE_string('ckpt_save_file_name', "competition.ckpt", """模板文件名""")
# tf.app.flags.DEFINE_string('output_graph', '/home/niusilong/work/AI/graph/competition-inception_v4_4860_crop4.pb', """Where to save the trained graph.""")


tf.app.flags.DEFINE_float('learning_rate_base', 0.0001, """初始学习率""")
tf.app.flags.DEFINE_float('learning_rate_decay', 0.99, """学习率的衰减率""")
tf.app.flags.DEFINE_integer('learning_rate_decay_steps', 100, """衰减步长""")
tf.app.flags.DEFINE_float('random_brightness', 0.2, """A percentage determining how much to randomly multiply the training image input pixels up or down by.""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result', """The name of the output classification layer in the retrained graph.""")
tf.app.flags.DEFINE_string('image_batch_name', 'image_batch', """The name of the output classification layer in the retrained graph.""")
tf.app.flags.DEFINE_string('accuracy_name', 'evaluation/accuracy', """The name of the output classification layer in the retrained graph.""")

CHECKPOINT_EXCLUDE_SCOPES = "InceptionV4/Logits,InceptionV4/AuxLogits"
# CHECKPOINT_EXCLUDE_SCOPES = "InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits"

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
#需要训练的网络层参数名称，在fine-tuning的过程中就是最后的全连接层，这里给出的是参数的前缀
TRAINABLE_SCOPE = "InceptionV4/Logits,InceptionV4/Auxlogits"
# TRAINABLE_SCOPE = "InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits"

#获取所有需要从从谷歌训练好的模型中加载的参数
def get_tuned_variables(variables_to_train=None, global_step=None):
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(",")]
    variable_to_restore = []
    #枚举inception_v3模型中所有的参数，然后判断是否需要从加载列表中移除
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variable_to_restore.append(var)
    if global_step != None:
        variable_to_restore.append(global_step)
    if variables_to_train != None:
        variable_to_restore.extend(variables_to_train)
    # print("variable_to_restore:",variable_to_restore)
    return variable_to_restore

#获取所有需要训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPE.split(",")]
    variables_to_train = []
    #枚举所有需要训练的参数前缀，并通过这些前缀找到所有的参数
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train
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
    
    boxes = tf.constant([[[0.07, 0.07, 0.93, 0.93]]])

    #从原始图像解析出像素矩阵，并根据图像尺寸还原图像
    decoded_image = tf.decode_raw(image, tf.float32)
    decoded_image = tf.reshape(decoded_image, shape=[FLAGS.image_size, FLAGS.image_size, 3])
    #定义神经网络输入层图片的大小
    distorted_image = change_image_util.preprocess_for_train_bbox(decoded_image, height=FLAGS.image_size, width=FLAGS.image_size, bbox=boxes, random_brightness=FLAGS.random_brightness)
    # distorted_image = decoded_image
    distorted_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])   #设置shape，否则shuffle_batch方法会报找不到shape错误
    #min_after_dequeue = 10000
    num_examples = 4500
    min_fraction_of_examples_in_queue = 0.4
    min_after_dequeue = int(num_examples * min_fraction_of_examples_in_queue)
    capacity = min_after_dequeue + 3 * FLAGS.batch_size
    num_threads = 6
    image_batch, label_batch, multi_hot_label_batch = tf.train.shuffle_batch([distorted_image, label, multi_hot_label], batch_size=FLAGS.batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=num_threads, name="image_batch")
    return image_batch, label_batch, multi_hot_label_batch

def main(argv=None):
    #加载预处理好的数据
    #定义inception_v3的输入,
    with tf.device('/cpu:0'):
        images, labels, multi_hot_labels = get_image_batch()
    
    
    # labels = tf.one_hot(labels, depth=5)
    #定义inception_v3模型，因为谷歌给出的只有模型参数取值，所以这里需要在这个代码中定义inception_v3的模型结构。虽然理论上需要区分训练和
    #测试中使用的模型，也就是说在测试时应该使用is_training=False, 但是因为预预先训练好的inception-v3模型中使用的batch normalization参数
    #与新的数据会有差异，导致结果很差，所以这里直接使用同一个模型来进行测试
    global_step = tf.Variable(0, trainable=False)
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits, _ = inception_v4.inception_v4(images, num_classes=FLAGS.num_classes, dropout_keep_prob=0.7)
    #获取需要训练的变量
    final_tensor = tf.nn.sigmoid(logits, name=FLAGS.final_tensor_name)
    tf.summary.histogram(FLAGS.final_tensor_name + '/activations', final_tensor)
    #定义交叉熵损失，注意在模型定义的时候已经将正则化损失加入损失集合了
    tf.losses.sigmoid_cross_entropy(multi_class_labels=multi_hot_labels, logits=logits, weights=1.0)

    #定义训练过程。这里minimize的过程中指定了需要优化的变量集合
    # learning_rate = LEARNING_RATE_BASE
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_base, global_step=global_step, decay_steps=FLAGS.learning_rate_decay_steps, decay_rate=FLAGS.learning_rate_decay)
    loss = tf.losses.get_total_loss()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.round(final_tensor), multi_hot_labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    #定义加载模型的函数
    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = ckpt.model_checkpoint_path
        trainable_variables = get_trainable_variables()
    else:
        #谷歌提供的训练好的模型文件地址
        ckpt_file = FLAGS.initial_ckpt_file
        trainable_variables = None
    # print("ckpt.model_checkpoint_path:",ckpt.model_checkpoint_path)
    print("ckpt_file:",ckpt_file)
    load_fn = slim.assign_from_checkpoint_fn(
        ckpt_file,
        get_tuned_variables(trainable_variables, global_step=global_step),
        ignore_missing_vars=True
    )

    #定义保存新的训练好的模型函数
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #初始化没有加载进来的变量。注意这个过程一定要在模型加载之前，否则初始化过程会将已经加载好的变量重新加载
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Loading tuned variables from %s' % ckpt_file)
        load_fn(sess)
        print("restored global_step:",global_step.eval())
        #保存graph
        for i in range(FLAGS.train_steps):
            #运行训练过程，这里不会更新全部的参数，只会更新指定的部分参数
            start_time = time.time()
            sess.run(train_step)
            duration = time.time() - start_time
            if (i+1) % 100 == 0 or (i+1) == FLAGS.train_steps:
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)
                
                step = global_step.eval()
                
                if step % 1000 == 0 or (i+1) == FLAGS.train_steps:
                    saver.save(sess=sess, save_path=os.path.join(FLAGS.ckpt_save_dir, FLAGS.ckpt_save_file_name), global_step=(step if (i+1) == FLAGS.train_steps or step % 1000 ==0 else None))
                    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [FLAGS.image_batch_name, FLAGS.final_tensor_name, FLAGS.accuracy_name])
                    with gfile.FastGFile(FLAGS.output_graph.replace(".pb", "_"+str(step)+".pb"), 'wb') as f:
                        f.write(output_graph_def.SerializeToString())
                validation_accuracy = sess.run(evaluation_step)
                loss_value = sess.run(loss)
                print("Step %d: Validation accuracy = %.2f%%, loss = %.5f (%.1f examples/sec; %.3f sec/batch)" % (step, validation_accuracy*100.0, loss_value, examples_per_sec, sec_per_batch))
        coord.request_stop()
        coord.join(threads)
        # print("evaluation_step.name:",evaluation_step.name)
        # print("multi_hot_labels.name：",multi_hot_labels.name)

if __name__ == '__main__':
    tf.app.run()