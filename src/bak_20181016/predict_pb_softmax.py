#coding=utf-8
'''使用最后预测值的平均数来预测'''
import tensorflow as tf
# import change_image_util
# import tensorflow.contrib.slim as slim
# import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
import numpy as np
from PIL import Image
# from tensorflow.python.framework import graph_util
# import matplotlib.pyplot as plt
import os
import re
# tf.app.flags.DEFINE_string('image_dir', "/home/niusilong/work/AI/images_test", """""")
tf.app.flags.DEFINE_string('image_dir', "/home/niusilong/projects/image-cut/src/main/webapp/images/120_3", """""")
tf.app.flags.DEFINE_integer('image_size', 192, """""")
tf.app.flags.DEFINE_integer('num_classes', 260, """""")
tf.app.flags.DEFINE_integer('batch_size', 30, """""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result', """The name of the output classification layer in the retrained graph.""")
tf.app.flags.DEFINE_string('graph_file', '../pb/competition_inception_v4_4740_gt30_batch30_34000.pb', """Where to save the trained graph.""")
tf.app.flags.DEFINE_integer('print_count', 10, """""")
tf.app.flags.DEFINE_float('min_score', 0.0046, """最低预测分数""")
tf.app.flags.DEFINE_integer('min_predict_count', 2, """最低预测分数""")
tf.app.flags.DEFINE_integer('max_predict_count', 5, """最低预测分数""")
tf.app.flags.DEFINE_string('predict_output_file_dir', "/home/niusilong/work/AI/predict", """最低预测分数""")
tf.app.flags.DEFINE_string('predict_output_file_name', "predict_result", """""")
tf.app.flags.DEFINE_integer('shuffle', 1, '''是否调整图片顺序 True:1, False:0''')
LABEL_DICT={1:"turn", 2:"land", 3:"albatross", 4:"f-16", 5:"bengal", 6:"shrubs", 7:"valley", 8:"mare", 9:"monastery", 10:"palace", 11:"grizzly", 12:"bear", 13:"flag", 14:"pillar", 15:"shops", 16:"canal", 17:"cars", 18:"doorway", 19:"sky", 20:"kit", 21:"vendor", 22:"interior", 23:"column", 24:"lake", 25:"buildings", 26:"baby", 27:"hawaii", 28:"road", 29:"birds", 30:"fan", 31:"cow", 32:"sheep", 33:"basket", 34:"marine", 35:"coral", 36:"dress", 37:"jet", 38:"roofs", 39:"white-tailed", 40:"forest", 41:"landscape", 42:"restaurant", 43:"shadows", 44:"cubs", 45:"close-up", 46:"gate", 47:"fly", 48:"ruins", 49:"stone", 50:"flight", 51:"sidewalk", 52:"store", 53:"tulip", 54:"sea", 55:"moose", 56:"tusks", 57:"oahu", 58:"clouds", 59:"temple", 60:"prop", 61:"tower", 62:"sails", 63:"caribou", 64:"stems", 65:"dunes", 66:"art", 67:"rocks", 68:"sunrise", 69:"tundra", 70:"silhouette", 71:"smoke", 72:"water", 73:"reptile", 74:"city", 75:"porcupine", 76:"woman", 77:"shore", 78:"balcony", 79:"clothes", 80:"mountain", 81:"tree", 82:"elephant", 83:"trunk", 84:"horizon", 85:"train", 86:"courtyard", 87:"vineyard", 88:"bulls", 89:"entrance", 90:"path", 91:"hillside", 92:"maui", 93:"sculpture", 94:"wall", 95:"face", 96:"food", 97:"fountain", 98:"window", 99:"sunset", 100:"peaks", 101:"tiger", 102:"snake", 103:"log", 104:"crab", 105:"branch", 106:"field", 107:"tails", 108:"canyon", 109:"cottage", 110:"slope", 111:"plane", 112:"petals", 113:"relief", 114:"mist", 115:"lynx", 116:"pyramid", 117:"palm", 118:"meadow", 119:"mosque", 120:"blooms", 121:"architecture", 122:"market", 123:"crystals", 124:"waves", 125:"lawn", 126:"ground", 127:"locomotive", 128:"lion", 129:"mule", 130:"den", 131:"coyote", 132:"ships", 133:"marsh", 134:"booby", 135:"kauai", 136:"hats", 137:"nets", 138:"anemone", 139:"fence", 140:"african", 141:"outside", 142:"cathedral", 143:"runway", 144:"vines", 145:"buddha", 146:"desert", 147:"bush", 148:"calf", 149:"crafts", 150:"indian", 151:"village", 152:"formula", 153:"squirrel", 154:"needles", 155:"formation", 156:"detail", 157:"buddhist", 158:"sign", 159:"lighthouse", 160:"foals", 161:"herd", 162:"costume", 163:"bridge", 164:"man", 165:"reefs", 166:"horns", 167:"night", 168:"reflection", 169:"cougar", 170:"light", 171:"fruit", 172:"horses", 173:"pool", 174:"zebra", 175:"street", 176:"vegetation", 177:"decoration", 178:"tables", 179:"terrace", 180:"statue", 181:"grass", 182:"sphinx", 183:"arctic", 184:"boats", 185:"coast", 186:"post", 187:"black", 188:"fish", 189:"skyline", 190:"head", 191:"windmills", 192:"giraffe", 193:"iguana", 194:"ice", 195:"polar", 196:"people", 197:"ceremony", 198:"church", 199:"castle", 200:"lizard", 201:"rodent", 202:"flowers", 203:"vehicle", 204:"beach", 205:"railroad", 206:"frost", 207:"door", 208:"antelope", 209:"house", 210:"snow", 211:"festival", 212:"scotland", 213:"pots", 214:"elk", 215:"cafe", 216:"sun", 217:"prototype", 218:"whales", 219:"fox", 220:"tracks", 221:"hut", 222:"harbor", 223:"plants", 224:"cat", 225:"glass", 226:"hills", 227:"barn", 228:"nest", 229:"cave", 230:"town", 231:"antlers", 232:"dock", 233:"truck", 234:"swimmers", 235:"garden", 236:"wood", 237:"butterfly", 238:"goat", 239:"stairs", 240:"monks", 241:"island", 242:"frozen", 243:"leaf", 244:"museum", 245:"cactus", 246:"ocean", 247:"hotel", 248:"girl", 249:"arch", 250:"monument", 251:"farms", 252:"park", 253:"dance", 254:"orchid", 255:"display", 256:"athlete", 257:"plaza", 258:"deer", 259:"sand", 260:"river"}
FLAGS = tf.app.flags.FLAGS
EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
print("graph_file: ",FLAGS.graph_file)
print("shuffle:",FLAGS.shuffle,", min_score:", FLAGS.min_score, ", min_predict_count:", FLAGS.min_predict_count,", max_predict_count:", FLAGS.max_predict_count)

with tf.gfile.FastGFile(FLAGS.graph_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
#获取所有需要从从谷歌训练好的模型中加载的参数
def get_multi_hot(labels):
    multi_hot = np.zeros(FLAGS.num_classes, dtype=np.float32)
    if labels == "":
        return multi_hot
    label_array = [int(x)-1 for x in labels.split(",")]
    for label in label_array:
        multi_hot[label] = 1.0
    return multi_hot
total_steps = -1
all_image_paths = []
array_index_array = []
def init_data():
    global total_steps
    global all_image_paths
    global array_index_array
    image_list = os.listdir(FLAGS.image_dir)
    image_list.sort()
    for image_name in image_list:
        if image_name.split(".")[-1] in EXTENSIONS:
            all_image_paths.append(os.path.join(FLAGS.image_dir, image_name))
    all_image_paths = sorted(all_image_paths, key=sort_img,reverse=False)
    array_index_array = np.arange(len(all_image_paths))
    if FLAGS.shuffle == 1:
        state = np.random.get_state()
        np.random.shuffle(all_image_paths)
        np.random.set_state(state)
        np.random.shuffle(array_index_array)
        array_index_array = array_index_array.argsort()
    # print("array_index_array:",array_index_array)
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
def sort_img(filename):
    try:
        img_val = int(filename.split("/")[-1].split(".")[0])
        return img_val
    except:
        return filename.split("/")[-1].split(".")[0]
def get_percentage(val):
    # print("np.shape(val):",np.shape(val))
    percent_val = np.array(val)
    # print(percent_val)
    sum_val = np.sum(val, axis=1)
    for i in range(len(val)):
        percent_val[i] = np.divide(percent_val[i], sum_val[i])
    # print("percent_val:",percent_val)
    # print("np.shape(percent_val):",np.shape(percent_val))
    return percent_val
def main(argv=None):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        init_data()
        # saver.restore(sess, FLAGS.ckpt_file)
        # print(global_step.eval())
        predict_array = []
        predict_array_mean = []
        changed_predict_array = []
        predict_output_array = []
        for step in range(total_steps):
            images, image_paths_val = get_image_data_array(step+1)
            final_tensor = sess.graph.get_tensor_by_name(FLAGS.final_tensor_name+':0')
            # print(sess.graph.get_tensor_by_name("image_batch:0").eval())
            final_tensor_val = sess.run(final_tensor, feed_dict={"image_batch:0": images})
            softmax_val = tf.nn.softmax(final_tensor_val).eval()
            # print("np.shape(final_tensor_val):", np.shape(final_tensor_val), ", np.shape(softmax_val):",np.shape(softmax_val))
            for i in range(len(final_tensor_val)):
                sorted_array = final_tensor_val[i].argsort()[::-1]
                prediction_result=[LABEL_DICT[label_index+1]+"("+str(label_index+1)+"):"+str(final_tensor_val[i][label_index]) for label_index in sorted_array[:FLAGS.print_count]]
                # print("np.sum(softmax_val[i])",np.sum(softmax_val[i]))
                predict_mean = np.mean(softmax_val[i], axis=0)
                prediction_result_mean=[LABEL_DICT[label_index+1]+"("+str(label_index+1)+"):"+str(softmax_val[i][label_index]) for label_index in sorted_array[:FLAGS.print_count]]
                predict_array.append(image_paths_val[i]+" "+str(prediction_result))
                predict_array_mean.append(image_paths_val[i]+" "+str(prediction_result_mean)+", predict_mean:"+str(predict_mean))
            for i in range(len(final_tensor_val)):
                sorted_array = softmax_val[i].argsort()[::-1]
                prediction_end_index = len(sorted_array)-1
                for j in range(len(sorted_array)):
                    if j < FLAGS.min_predict_count: continue
                    if softmax_val[i][sorted_array[j]] < FLAGS.min_score or j >= FLAGS.max_predict_count:
                        prediction_end_index = j
                        break
                prediction_result=[LABEL_DICT[label_index+1]+"("+str(label_index+1)+"):"+str(softmax_val[i][label_index]) for label_index in sorted_array[:prediction_end_index]]
                prediction_output_result = [label_index+1 for label_index in sorted_array[:prediction_end_index]]
                prediction_output_result.sort()
                changed_predict_array.append(image_paths_val[i]+" "+str(prediction_result))
                predict_output_array.append(image_paths_val[i].split("/")[-1].split(".")[0]+" "+re.sub("['\\[\\]\\s]","",str(prediction_output_result)))
        print("******************************************************************")
        for i in range(len(array_index_array)):
            print(predict_array[array_index_array[i]])
            print(predict_array_mean[array_index_array[i]])
        print("******************************************************************")
        for i in range(len(array_index_array)):
            print(changed_predict_array[array_index_array[i]])
        print("******************************************************************")
        #导出预测文件
        files = os.listdir(FLAGS.predict_output_file_dir)
        count = 0
        for i in range(len(files)):
            if files[i].startswith(FLAGS.predict_output_file_name):
                count += 1
        export_file_path = os.path.join(FLAGS.predict_output_file_dir, FLAGS.predict_output_file_name+"_"+str(count+1)+".txt")
        print("predict_file:",export_file_path)
        with tf.gfile.GFile(export_file_path, "wb") as f:
            for i in range(len(array_index_array)):
                f.write(predict_output_array[array_index_array[i]]+"\n")
        print("export file over!")
if __name__ == '__main__':
    tf.app.run()