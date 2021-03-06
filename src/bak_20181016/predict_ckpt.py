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
from PIL import Image
import matplotlib.pyplot as plt
#处理好之后的文件
#保存训练好之后的路径，这里可以将使用新数据训练得到的完整模型保存下来，如果计算资源充足，还可以在训练完最后的全连接层之后再训练所有网络层，这样可以使得新模型更加贴近数据

FLAGS = tf.app.flags.FLAGS
#自定义参数，可以在sh脚本文件中修改
tf.app.flags.DEFINE_integer('train_steps', 1000, """""")
tf.app.flags.DEFINE_integer('image_size', 192, """""")
tf.app.flags.DEFINE_integer('batch_size', 30, """""")
tf.app.flags.DEFINE_integer('num_classes', 260, """""")
# tf.app.flags.DEFINE_string('ckpt_save_file_path', "/home/niusilong/work/AI/ckpt/multi-label-competition/multi-label-competition.ckpt", """模板文件名""")
tf.app.flags.DEFINE_string('ckpt_save_file_path', "/home/niusilong/work/AI/ckpt/competition-inception_v4_192/competition.ckpt", """模板文件名""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result', """The name of the output classification layer in the retrained graph.""")
tf.app.flags.DEFINE_string('image_dir', "/home/niusilong/work/AI/images_test", """""")
tf.app.flags.DEFINE_integer('print_count', 10, """""")
LABEL_DICT={1:"turn", 2:"land", 3:"albatross", 4:"f-16", 5:"bengal", 6:"shrubs", 7:"valley", 8:"mare", 9:"monastery", 10:"palace", 11:"grizzly", 12:"bear", 13:"flag", 14:"pillar", 15:"shops", 16:"canal", 17:"cars", 18:"doorway", 19:"sky", 20:"kit", 21:"vendor", 22:"interior", 23:"column", 24:"lake", 25:"buildings", 26:"baby", 27:"hawaii", 28:"road", 29:"birds", 30:"fan", 31:"cow", 32:"sheep", 33:"basket", 34:"marine", 35:"coral", 36:"dress", 37:"jet", 38:"roofs", 39:"white-tailed", 40:"forest", 41:"landscape", 42:"restaurant", 43:"shadows", 44:"cubs", 45:"close-up", 46:"gate", 47:"fly", 48:"ruins", 49:"stone", 50:"flight", 51:"sidewalk", 52:"store", 53:"tulip", 54:"sea", 55:"moose", 56:"tusks", 57:"oahu", 58:"clouds", 59:"temple", 60:"prop", 61:"tower", 62:"sails", 63:"caribou", 64:"stems", 65:"dunes", 66:"art", 67:"rocks", 68:"sunrise", 69:"tundra", 70:"silhouette", 71:"smoke", 72:"water", 73:"reptile", 74:"city", 75:"porcupine", 76:"woman", 77:"shore", 78:"balcony", 79:"clothes", 80:"mountain", 81:"tree", 82:"elephant", 83:"trunk", 84:"horizon", 85:"train", 86:"courtyard", 87:"vineyard", 88:"bulls", 89:"entrance", 90:"path", 91:"hillside", 92:"maui", 93:"sculpture", 94:"wall", 95:"face", 96:"food", 97:"fountain", 98:"window", 99:"sunset", 100:"peaks", 101:"tiger", 102:"snake", 103:"log", 104:"crab", 105:"branch", 106:"field", 107:"tails", 108:"canyon", 109:"cottage", 110:"slope", 111:"plane", 112:"petals", 113:"relief", 114:"mist", 115:"lynx", 116:"pyramid", 117:"palm", 118:"meadow", 119:"mosque", 120:"blooms", 121:"architecture", 122:"market", 123:"crystals", 124:"waves", 125:"lawn", 126:"ground", 127:"locomotive", 128:"lion", 129:"mule", 130:"den", 131:"coyote", 132:"ships", 133:"marsh", 134:"booby", 135:"kauai", 136:"hats", 137:"nets", 138:"anemone", 139:"fence", 140:"african", 141:"outside", 142:"cathedral", 143:"runway", 144:"vines", 145:"buddha", 146:"desert", 147:"bush", 148:"calf", 149:"crafts", 150:"indian", 151:"village", 152:"formula", 153:"squirrel", 154:"needles", 155:"formation", 156:"detail", 157:"buddhist", 158:"sign", 159:"lighthouse", 160:"foals", 161:"herd", 162:"costume", 163:"bridge", 164:"man", 165:"reefs", 166:"horns", 167:"night", 168:"reflection", 169:"cougar", 170:"light", 171:"fruit", 172:"horses", 173:"pool", 174:"zebra", 175:"street", 176:"vegetation", 177:"decoration", 178:"tables", 179:"terrace", 180:"statue", 181:"grass", 182:"sphinx", 183:"arctic", 184:"boats", 185:"coast", 186:"post", 187:"black", 188:"fish", 189:"skyline", 190:"head", 191:"windmills", 192:"giraffe", 193:"iguana", 194:"ice", 195:"polar", 196:"people", 197:"ceremony", 198:"church", 199:"castle", 200:"lizard", 201:"rodent", 202:"flowers", 203:"vehicle", 204:"beach", 205:"railroad", 206:"frost", 207:"door", 208:"antelope", 209:"house", 210:"snow", 211:"festival", 212:"scotland", 213:"pots", 214:"elk", 215:"cafe", 216:"sun", 217:"prototype", 218:"whales", 219:"fox", 220:"tracks", 221:"hut", 222:"harbor", 223:"plants", 224:"cat", 225:"glass", 226:"hills", 227:"barn", 228:"nest", 229:"cave", 230:"town", 231:"antlers", 232:"dock", 233:"truck", 234:"swimmers", 235:"garden", 236:"wood", 237:"butterfly", 238:"goat", 239:"stairs", 240:"monks", 241:"island", 242:"frozen", 243:"leaf", 244:"museum", 245:"cactus", 246:"ocean", 247:"hotel", 248:"girl", 249:"arch", 250:"monument", 251:"farms", 252:"park", 253:"dance", 254:"orchid", 255:"display", 256:"athlete", 257:"plaza", 258:"deer", 259:"sand", 260:"river"}
#需要训练的网络层参数名称，在fine-tuning的过程中就是最后的全连接层，这里给出的是参数的前缀
EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
total_steps = -1
all_image_paths = []
def init_data():
    global total_steps
    global all_image_paths
    image_list = os.listdir(FLAGS.image_dir)
    image_list.sort()
    for image_name in image_list:
        if image_name.split(".")[-1] in EXTENSIONS:
            all_image_paths.append(os.path.join(FLAGS.image_dir, image_name))
    total_steps = int(len(all_image_paths) / FLAGS.batch_size) if len(all_image_paths) % FLAGS.batch_size == 0 else (int(len(all_image_paths) / FLAGS.batch_size)+1)
    print("total_steps:",total_steps, ", len(all_image_paths):",len(all_image_paths))
def get_image_data_array(step):
    image_array = []
    image_paths = []
    loop_lines = []
    start_index = (step-1)*FLAGS.batch_size
    end_index = step*FLAGS.batch_size
    while(True):
        if(len(loop_lines) >= step*FLAGS.batch_size):
            selected_paths = loop_lines[start_index:end_index]
            break
        else:
            loop_lines.extend(all_image_paths)
    for path in selected_paths:
        im = Image.open(path)
        im = im.resize((FLAGS.image_size, FLAGS.image_size))
        im_array = np.array(im, dtype=np.float32)
        # plt.imshow(np.asarray(im_array, dtype=np.uint8))
        # plt.show()
        # image = tf.reshape(image, shape=[FLAGS.image_size, FLAGS.image_size, 3])
        image_array.append(im_array)
        image_paths.append(path)
    return image_array, image_paths
def get_multi_hot(labels):
    multi_hot = np.zeros(FLAGS.num_classes, dtype=np.float32)
    if labels == "":
        return multi_hot
    label_array = [int(x)-1 for x in labels.split(",")]
    for label in label_array:
        multi_hot[label] = 1.0
    return multi_hot
def main(argv=None):
    #加载预处理好的数据
    #定义inception_v4的输入,
    init_data()
    images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
    multi_hot_labels = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.num_classes])
    # labels = tf.one_hot(labels, depth=5)
    #定义inception_v4模型，因为谷歌给出的只有模型参数取值，所以这里需要在这个代码中定义inception_v4的模型结构。虽然理论上需要区分训练和
    #测试中使用的模型，也就是说在测试时应该使用is_training=False, 但是因为预预先训练好的inception-v3模型中使用的batch normalization参数
    #与新的数据会有差异，导致结果很差，所以这里直接使用同一个模型来进行测试
    global_step = tf.Variable(0, trainable=False)
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits, _ = inception_v4.inception_v4(images, num_classes=FLAGS.num_classes, dropout_keep_prob=1.0)
    #获取需要训练的变量
    final_tensor = tf.nn.sigmoid(logits, name=FLAGS.final_tensor_name)

    #定义保存新的训练好的模型函数
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #初始化没有加载进来的变量。注意这个过程一定要在模型加载之前，否则初始化过程会将已经加载好的变量重新加载
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, FLAGS.ckpt_save_file_path)
        print("restored global_step:",global_step.eval())
        predict_array = []
        for step in range(total_steps):
            images_val, image_paths_val = get_image_data_array(step+1)
            # print(sess.graph.get_tensor_by_name("image_batch:0").eval())
            final_tensor_val = sess.run(final_tensor, feed_dict={images:images_val})
            for i in range(len(final_tensor_val)):
                sorted_array = final_tensor_val[i].argsort()[::-1]
                prediction_result=[LABEL_DICT[label_index+1]+"("+str(label_index+1)+"):"+str(final_tensor_val[i][label_index]) for label_index in sorted_array[:FLAGS.print_count]]
                predict_array.append(image_paths_val[i]+" "+str(prediction_result))
        for i in range(len(all_image_paths)):
            print(predict_array[i])
if __name__ == '__main__':
    tf.app.run()