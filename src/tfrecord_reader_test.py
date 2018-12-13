#coding=utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import ast
import json
from array import array
IMAGE_SIZE = 120
TOTAL_IMAGES = 4500
#使用tf.train.match_filename_once来获取文件列表
# files = tf.train.match_filenames_once("/home/niusilong/work/AI/flowers/flower_photos_part_0.tfrecords")

tf.app.flags.DEFINE_string('tfrecord_file_pattern', "/home/niusilong/work/AI/tfrecords/competition_part_*.tfrecords", """包含图片，标签，multi-hot数据的压缩包""")
FLAGS = tf.app.flags.FLAGS
LABEL_DICT={1:"turn", 2:"land", 3:"albatross", 4:"f-16", 5:"bengal", 6:"shrubs", 7:"valley", 8:"mare", 9:"monastery", 10:"palace", 11:"grizzly", 12:"bear", 13:"flag", 14:"pillar", 15:"shops", 16:"canal", 17:"cars", 18:"doorway", 19:"sky", 20:"kit", 21:"vendor", 22:"interior", 23:"column", 24:"lake", 25:"buildings", 26:"baby", 27:"hawaii", 28:"road", 29:"birds", 30:"fan", 31:"cow", 32:"sheep", 33:"basket", 34:"marine", 35:"coral", 36:"dress", 37:"jet", 38:"roofs", 39:"white-tailed", 40:"forest", 41:"landscape", 42:"restaurant", 43:"shadows", 44:"cubs", 45:"close-up", 46:"gate", 47:"fly", 48:"ruins", 49:"stone", 50:"flight", 51:"sidewalk", 52:"store", 53:"tulip", 54:"sea", 55:"moose", 56:"tusks", 57:"oahu", 58:"clouds", 59:"temple", 60:"prop", 61:"tower", 62:"sails", 63:"caribou", 64:"stems", 65:"dunes", 66:"art", 67:"rocks", 68:"sunrise", 69:"tundra", 70:"silhouette", 71:"smoke", 72:"water", 73:"reptile", 74:"city", 75:"porcupine", 76:"woman", 77:"shore", 78:"balcony", 79:"clothes", 80:"mountain", 81:"tree", 82:"elephant", 83:"trunk", 84:"horizon", 85:"train", 86:"courtyard", 87:"vineyard", 88:"bulls", 89:"entrance", 90:"path", 91:"hillside", 92:"maui", 93:"sculpture", 94:"wall", 95:"face", 96:"food", 97:"fountain", 98:"window", 99:"sunset", 100:"peaks", 101:"tiger", 102:"snake", 103:"log", 104:"crab", 105:"branch", 106:"field", 107:"tails", 108:"canyon", 109:"cottage", 110:"slope", 111:"plane", 112:"petals", 113:"relief", 114:"mist", 115:"lynx", 116:"pyramid", 117:"palm", 118:"meadow", 119:"mosque", 120:"blooms", 121:"architecture", 122:"market", 123:"crystals", 124:"waves", 125:"lawn", 126:"ground", 127:"locomotive", 128:"lion", 129:"mule", 130:"den", 131:"coyote", 132:"ships", 133:"marsh", 134:"booby", 135:"kauai", 136:"hats", 137:"nets", 138:"anemone", 139:"fence", 140:"african", 141:"outside", 142:"cathedral", 143:"runway", 144:"vines", 145:"buddha", 146:"desert", 147:"bush", 148:"calf", 149:"crafts", 150:"indian", 151:"village", 152:"formula", 153:"squirrel", 154:"needles", 155:"formation", 156:"detail", 157:"buddhist", 158:"sign", 159:"lighthouse", 160:"foals", 161:"herd", 162:"costume", 163:"bridge", 164:"man", 165:"reefs", 166:"horns", 167:"night", 168:"reflection", 169:"cougar", 170:"light", 171:"fruit", 172:"horses", 173:"pool", 174:"zebra", 175:"street", 176:"vegetation", 177:"decoration", 178:"tables", 179:"terrace", 180:"statue", 181:"grass", 182:"sphinx", 183:"arctic", 184:"boats", 185:"coast", 186:"post", 187:"black", 188:"fish", 189:"skyline", 190:"head", 191:"windmills", 192:"giraffe", 193:"iguana", 194:"ice", 195:"polar", 196:"people", 197:"ceremony", 198:"church", 199:"castle", 200:"lizard", 201:"rodent", 202:"flowers", 203:"vehicle", 204:"beach", 205:"railroad", 206:"frost", 207:"door", 208:"antelope", 209:"house", 210:"snow", 211:"festival", 212:"scotland", 213:"pots", 214:"elk", 215:"cafe", 216:"sun", 217:"prototype", 218:"whales", 219:"fox", 220:"tracks", 221:"hut", 222:"harbor", 223:"plants", 224:"cat", 225:"glass", 226:"hills", 227:"barn", 228:"nest", 229:"cave", 230:"town", 231:"antlers", 232:"dock", 233:"truck", 234:"swimmers", 235:"garden", 236:"wood", 237:"butterfly", 238:"goat", 239:"stairs", 240:"monks", 241:"island", 242:"frozen", 243:"leaf", 244:"museum", 245:"cactus", 246:"ocean", 247:"hotel", 248:"girl", 249:"arch", 250:"monument", 251:"farms", 252:"park", 253:"dance", 254:"orchid", 255:"display", 256:"athlete", 257:"plaza", 258:"deer", 259:"sand", 260:"river"}
files = tf.train.match_filenames_once(FLAGS.tfrecord_file_pattern)
filename_queue = tf.train.string_input_producer(files, shuffle=False, num_epochs=3)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
        'multi_hot_label': tf.FixedLenFeature([260], tf.float32),
        'image_path': tf.FixedLenFeature([], tf.string)
    }
)
image = tf.decode_raw(features['image'], tf.float32)
label = features['label']
multi_hot_label = features['multi_hot_label']
image_path = features['image_path']

with tf.Session() as sess:
    #虽然在本段程序中没有声明任何变量，但使用tf.train.match_filienames_once函数时需要初始化一些变量
    tf.local_variables_initializer().run()
    print(files.eval())
    #声明tf.train.Coordinator类来协同不同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #多次执行获取数据的操作
    for i in range(TOTAL_IMAGES):
        if i >= 10:
            image.eval()
            continue
        image = tf.reshape(image, shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
        image_eval = image.eval()
        # print("image_eval:",image_eval)
        # print("image.eval(): ", str(image.eval()).replace("\n",""))
        try:
            plt.imshow(image_eval)
            plt.show()
            # print(label.eval(), ",", str(image.eval()).replace("\n", ""))
        except tf.errors.OutOfRangeError as e:
            break
    #多次执行获取数据的操作
    for i in range(TOTAL_IMAGES):
        if i >= 10:
            label.eval()
            continue
        try:
            label_array = str(label.eval(), encoding = "utf-8").split(",")
            display_label_array = ["["+x+"]"+LABEL_DICT[int(x)] for x in label_array]
            print("i: %d, label: %s" % (i, display_label_array))
            # print("i:", i, ", image_path:", image_path.eval())
        except tf.errors.OutOfRangeError as e:
            break
    for i in range(TOTAL_IMAGES):
        if i >= 10:
            multi_hot_label.eval()
            continue
        try:
            print("i: %d, multi_hot_label: %s" % (i, multi_hot_label.eval()))
            # print("i:", i, ", image_path:", image_path.eval())
        except tf.errors.OutOfRangeError as e:
            break
    coord.request_stop()
    coord.join()