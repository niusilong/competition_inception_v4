import tensorflow as tf
import matplotlib.pyplot as plt
import os
import threading
import time
from PIL import Image
# LABEL_FILE = "/home/niusilong/projects/image-cut/src/main/webapp/label_flower_photos_id_order.txt"
# CROPED_IMAGE_SAVE_DIR = "/home/niusilong/projects/image-cut/src/main/webapp/flower_photos_cropped"

LABEL_FILE = "/home/niusilong/projects/image-cut/src/main/resources/label_file/3_tag_absolute_path.txt"
CROPED_IMAGE_SAVE_DIR = "/home/niusilong/projects/image-cut/src/main/webapp/images/crop9_120_3"

IMAGE_SIZE = 360
THREAD_NUM = 1

single_width = int(IMAGE_SIZE/3)
single_height = int(IMAGE_SIZE/3)
tf_bound_array = []   #[top,left,width, height]
tf_bound_array.append([0, 0, single_height, single_width])
tf_bound_array.append([0, single_width, single_height, single_width])
tf_bound_array.append([0, 2*single_width, single_height, single_width])
tf_bound_array.append([single_height, 0, single_height, single_width])
tf_bound_array.append([single_height, single_width, single_height, single_width])
tf_bound_array.append([single_height, 2*single_width, single_height, single_width])
tf_bound_array.append([2*single_height, 0, single_height, single_width])
tf_bound_array.append([2*single_height, single_width, single_height, single_width])
tf_bound_array.append([2*single_height, 2*single_width, single_height, single_width])

pil_bound_array = []  #(left, upper, right, lower)-tuple
pil_bound_array.append([0, 0, single_width, single_height])
pil_bound_array.append([single_width, 0, 2*single_width, single_height])
pil_bound_array.append([2*single_width, 0, 3*single_width, single_height])
pil_bound_array.append([0, single_height, single_width, 2*single_height])
pil_bound_array.append([single_width, single_height, 2*single_width, 2*single_height])
pil_bound_array.append([2*single_width, single_height, 3*single_width, 2*single_height])
pil_bound_array.append([0, 2*single_height, single_width, 3*single_height])
pil_bound_array.append([single_width, 2*single_height, 2*single_width, 3*single_height])
pil_bound_array.append([2*single_width, 2*single_height, 3*single_width, 3*single_height])


count = 0
def action_plt(sep_lines):
    # time.sleep(1)
    print('sub thread start!the thread name is:%s\r' % threading.currentThread().getName())
    print("len(sep_lines):",len(sep_lines))
    global count
    with tf.Session() as sess:
        for i in range(len(sep_lines)):
            line_array = sep_lines[i].split(" ")
            image_raw_data = tf.gfile.FastGFile(line_array[0], 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # print(tf.shape(image).eval())
            image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
            # plt.imshow(image.eval())
            # plt.show()
            path_array = line_array[0].split("/")
            file_name = path_array[-1][0:path_array[-1].rfind(".")]
            file_suffix = path_array[-1].split(".")[-1]
            # print("file_name:%s, file_suffix:%s" % (file_name, file_suffix))
            for j in range(len(tf_bound_array)):
                save_dir = os.path.join(CROPED_IMAGE_SAVE_DIR, path_array[-2])
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                cropped_file_path = os.path.join(save_dir, file_name+"_"+str(j)+"."+file_suffix)
                if os.path.exists(cropped_file_path):
                    continue
                croped_image = tf.image.crop_to_bounding_box(image, tf_bound_array[j][0], tf_bound_array[j][1], tf_bound_array[j][2], tf_bound_array[j][3])
                plt.imshow(croped_image.eval())
                plt.show()
            count += 1
            print("count:",count)
def action_pil(sep_lines):
    # time.sleep(1)
    print('sub thread start!the thread name is:%s\r' % threading.currentThread().getName())
    print("len(sep_lines):",len(sep_lines))
    global count
    with tf.Session() as sess:
        for i in range(len(sep_lines)):
            line_array = sep_lines[i].strip().split(" ")
            path_array = line_array[0].split("/")
            file_name = path_array[-1][0:path_array[-1].rfind(".")]
            file_suffix = path_array[-1].split(".")[-1]
            im = Image.open(line_array[0])
            im = im.resize((IMAGE_SIZE, IMAGE_SIZE))
            # im.show()
            save_dir = os.path.join(CROPED_IMAGE_SAVE_DIR)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for j in range(len(pil_bound_array)):
                cropped_file_path = os.path.join(save_dir, file_name+"_"+str(j)+"."+file_suffix)
                if os.path.exists(cropped_file_path):
                    continue
                #设置要裁剪的区域
                region=im.crop(pil_bound_array[j]) #此时，region是一个新的图像对象。
                # region.show()
                region.save(cropped_file_path, "JPEG")
            count += 1
            print("count:",count)
def main(args):
    global count
    lines = open(LABEL_FILE).readlines()
    sep_line_batch = []
    nums_per_thread = int(len(lines)/THREAD_NUM)
    for i in range(len(lines)):
        if int(i / nums_per_thread) > len(sep_line_batch)-1 and len(sep_line_batch) < THREAD_NUM:
            sep_line = []
            sep_line_batch.append(sep_line)
        sep_line_batch[-1].append(lines[i]);
    for i in range(len(sep_line_batch)):
        t =threading.Thread(target=action_pil,args=(sep_line_batch[i],))
        t.start()
if __name__ == '__main__':
    tf.app.run()

