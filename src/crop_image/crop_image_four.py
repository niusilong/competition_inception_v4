import tensorflow as tf
import matplotlib.pyplot as plt
import os
import threading
import time
from PIL import Image
# LABEL_FILE = "/home/niusilong/projects/image-cut/src/main/webapp/label_flower_photos_id_order.txt"
# CROPED_IMAGE_SAVE_DIR = "/home/niusilong/projects/image-cut/src/main/webapp/flower_photos_cropped"

IMAGE_DIR = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/webapp/images/120_4"
DEFAULT_CROPED_IMAGE_SAVE_DIR = "/home/niusilong/svn/svn_repository_ai/image-cut/src/main/webapp/images/crop4_120_4_128"

IMAGE_SIZE = int(192)
print("IMAGE_SIZE:",IMAGE_SIZE)
THREAD_NUM = 5

single_width = int(IMAGE_SIZE/3)
single_height = int(IMAGE_SIZE/3)
pil_bound_array = []  #(left, upper, right, lower)-tuple
pil_bound_array.append([0, 0, 2*single_width, 2*single_height])
pil_bound_array.append([single_width, 0, 3*single_width, 2*single_height])
pil_bound_array.append([0, single_height, 2*single_width, 3*single_height])
pil_bound_array.append([single_width, single_height, 3*single_width, 3*single_height])


count = 0
def action_pil(img_paths, croped_image_save_dir):
    if not os.path.exists(croped_image_save_dir):
        os.mkdir(croped_image_save_dir)
    # time.sleep(1)
    print('sub thread start!the thread name is:%s\r' % threading.currentThread().getName())
    print("len(img_paths):",len(img_paths))
    global count
    with tf.Session() as sess:
        for img_path in img_paths:
            img_file_name = img_path.split("/")[-1]
            print("img_file_name:",img_file_name)
            file_name = img_file_name[0:img_file_name.rfind(".")]
            file_suffix = img_file_name.split(".")[-1]
            im = Image.open(img_path)
            im = im.resize((IMAGE_SIZE, IMAGE_SIZE))
            # im.show()
            # save_dir = os.path.join(CROPED_IMAGE_SAVE_DIR)
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)

            for j in range(len(pil_bound_array)):
                print("file_name:",file_name)
                cropped_file_path = os.path.join(croped_image_save_dir, file_name+"_"+str(j)+"."+file_suffix)
                print("cropped_file_path:",cropped_file_path)
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
    images = os.listdir(IMAGE_DIR)
    for i in range(len(images)):
        images[i] = os.path.join(IMAGE_DIR, images[i])
    print("images:",images)
    sep_line_batch = []
    nums_per_thread = int(len(images)/THREAD_NUM)
    for i in range(len(images)):
        if int(i / nums_per_thread) > len(sep_line_batch)-1 and len(sep_line_batch) < THREAD_NUM:
            sep_line = []
            sep_line_batch.append(sep_line)
        sep_line_batch[-1].append(images[i]);
    for i in range(len(sep_line_batch)):
        t =threading.Thread(target=action_pil,args=(sep_line_batch[i], DEFAULT_CROPED_IMAGE_SAVE_DIR, ))
        t.start()
if __name__ == '__main__':
    tf.app.run()

