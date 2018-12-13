#coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape

def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)    #调整亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)   #调整饱和度
        image = tf.image.random_hue(image, max_delta=0.2)   #调整色调
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train_bbox(image, height, width, bbox, random_brightness):
    #转换图像张量的数型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    if bbox != None:   
        #随机截取图像，减小需要关注的物体大小对图像识别算法的影响
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox, min_object_covered=0.80)
        # print("tf.shape(image).eval(): ", tf.shape(image).eval())
        distorted_image = tf.slice(image, bbox_begin, bbox_size)
        #将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的
        distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    else:
        distorted_image = image 
    
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(1))
    # return distorted_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(distorted_image, brightness_value)
    # distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return brightened_image

def preprocess_for_train(image, height, width, bbox, random_brightness):
    #如果没有标注框，则认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1,1,4])
    #转换图像张量的数型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    # print("tf.shape(image).eval(): ", tf.shape(image).eval())
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    #将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(1))
    # return distorted_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(distorted_image, brightness_value)
    # distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return brightened_image
def preprocess_for_train_without_crop(image, height, width, bbox):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的
    distorted_image = tf.image.resize_images(image, [height, width], method=np.random.randint(4))
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(1))
    return distorted_image
'''
image_raw_data = tf.gfile.FastGFile("/home/niusilong/work/AI/training/image/1000.jpeg", "rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    #运行六次获得6种不同的图像
    for i in range(6):
        #将图像的尺寸调整为299*299
        result = preprocess_for_train(img_data, 299, 299, boxes)
        print(tf.shape(result).eval())
        plt.imshow(result.eval())
        plt.show()
'''
if __name__ == '__main__':
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1,1,4])
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box([150, 150, 3], bounding_boxes=bbox)
    with tf.Session() as sess:
        print(bbox_begin.eval())
        print(bbox_size.eval())
