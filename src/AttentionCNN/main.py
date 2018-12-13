#!/usr/bin/env python
# coding=utf-8
import sys
import os
import numpy as np
import tensorflow as tf
import cv2
from model import CaptionGenerator
import random
import re
from tensorflow.python.framework import ops
from tensorflow.python import pywrap_tensorflow
from PIL import Image, ImageEnhance
import threading

imgIndex = 0
imageNameLst = []
imageLabelLst = []
imgDir = 'image'
MAX_LABEL = 262
MAX_TIME = 7
batchSize = 30

def randomCrop(image):
    """
    瀵瑰浘鍍忛殢鎰忓壀鍒�,鑰冭檻鍒板浘鍍忓ぇ灏忚寖鍥�(68,68),浣跨敤涓€涓竴涓ぇ浜�(36*36)鐨勭獥鍙ｈ繘琛屾埅鍥�
    :param image: PIL鐨勫浘鍍廼mage
    :return: 鍓垏涔嬪悗鐨勫浘鍍�
    """
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_size_width = np.random.randint(image.size[0]-30, image.size[0])
    crop_win_size_height = np.random.randint(image.size[1]-30, image.size[1])
    random_region = (
        (image_width - crop_win_size_width) >> 1, (image_height - crop_win_size_height) >> 1, (image_width + crop_win_size_width) >> 1,
        (image_height + crop_win_size_height) >> 1)
    return image.crop(random_region)

def randomColor(image):
    """
    瀵瑰浘鍍忚繘琛岄鑹叉姈鍔�
    :param image: PIL鐨勫浘鍍廼mage
    :return: 鏈夐鑹茶壊宸殑鍥惧儚image
    """
    random_factor = np.random.randint(4, 19) / 10.  # 闅忔満鍥犲瓙
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 璋冩暣鍥惧儚鐨勯ケ鍜屽害
    random_factor = np.random.randint(10, 15) / 10.  # 闅忔満鍥犲瓙
    contrast_image = ImageEnhance.Contrast(color_image).enhance(random_factor)  # 璋冩暣鍥惧儚瀵规瘮搴�
    random_factor = np.random.randint(0, 13) / 10.  # 闅忔満鍥犲瓙
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 璋冩暣鍥惧儚閿愬害

def preprocess(imagePath):
    im = Image.open(imagePath)
    image_width = im.size[0]
    image_height = im.size[1]
    im = randomCrop(im)
    im = randomColor(im).resize((299, 299))
    return np.array(im)

# im = preprocess("image/1020.jpeg")
# cv2.imwrite('test.jpg', im)

labelFile = open('train_tagId.txt')
lines = labelFile.readlines()
wholeImages = []
wholeLabels = []

lock = threading.Lock()
def myThread(lines):

    for line in lines:
        name = 'image/'+line.strip().split(' ')[0]+'.jpeg'

        im = preprocess(name)
        im = im.astype(np.float32)
        #im = (im.astype(np.float32)/255.0 - 0.5)*2  #[0, 255] scale to [-1, 1]

        labels = []
        for index in line.split(' ')[1].split(','):
            labels.append(int(index))

        lock.acquire()
        wholeImages.append(im)
        wholeLabels.append(labels)
        lock.release()


imgDistortCount = 1
threads = []
for i in range(imgDistortCount):
    t1 = threading.Thread(target=myThread, args=(lines,))
    t1.start()
    threads.append(t1)

for i in range(imgDistortCount):
    threads[i].join()

wholeData = list(zip(wholeImages, wholeLabels))   #image[192, 192, 3] ==> ontHot label [0,0, ...0 ,1, 0,0,0,...]
index = 0
def get_image_batch():  #images [n, 192,192,3] ==> labels [[0,0,1,0,0,...], [1,0,0,...], ...]
    global index
    if(index+batchSize>len(wholeImages)):
        random.shuffle(wholeData)
        index = 0

    wimages, wlabels = zip(*wholeData)
    index += batchSize

    rimages = []
    for im in wimages[index-batchSize:index]:
        rimages.append(im)

    rlabels = []
    for lb in wlabels[index-batchSize:index]:
        labels = np.ones((MAX_TIME))*(MAX_LABEL-1.0)
        labels[0] = 0
        i = 1
        random.shuffle(lb)
        for lbidx in lb:
            labels[i] = lbidx
            i += 1
        rlabels.append(labels)

    return (rimages, rlabels)

'''
for i in range(800):
    get_image_batch()

file = open('random.txt', 'w')
for i in range(10):

    img, lb = get_image_batch()
    for k in range(40):
        print(str(list(lb[k][1:]).index(MAX_LABEL-1.0))+' \n')

        if(list(lb[k][1:]).index(MAX_LABEL-1.0)<=2):
            cv2.imwrite('step/'+str(i)+'.jpg', ((img[k]/2+0.5)*255).astype(np.uint8))
            for j in lb[k]:
                file.write(str(int(j))+',')
            file.write('\r\n')
            break
'''

    # print (lb[1])
    # cv2.imshow('test', ((img[1]/2+0.5)*255).astype(np.uint8))
    # cv2.waitKey(0)

    #with tf.Graph().as_default():
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=30, intra_op_parallelism_threads=30)) as sess:

        restore_vars = []
        generator = CaptionGenerator(batchSize)
        _, cost = generator.build_model()
        for var in tf.global_variables():
            if('InceptionV4' in var.name):
               restore_vars.append(var)
            #if('Adam' not in var.name):
            #   restore_vars.append(var)

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Attention")

        gStep = tf.Variable(tf.constant(0))
        learning_rate = tf.train.exponential_decay(0.002, gStep, 600, 0.92, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, vars = zip(*optimizer.compute_gradients(cost, var_list=train_vars))  #var_list=train_vars
        gradients, _ = tf.clip_by_global_norm(gradients, 1000)
        train_op = optimizer.apply_gradients(zip(gradients, vars))

        sess.run(tf.global_variables_initializer())
        checkpoint_dir = 'step'
        saver = tf.train.Saver(var_list=restore_vars)    #var_list=restore_vars
        saver.restore(sess, 'step/competition.ckpt-24000')
        saver = tf.train.Saver()

        # writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        for epoch in range(1000000):
            images, labels = get_image_batch()
            feed_dict = {generator.images: images, generator.captions: labels, gStep: epoch}
            _, lr, loss = sess.run([train_op, learning_rate, cost], feed_dict=feed_dict)
            print(str(epoch) + ' lr: ' + str(lr) + ' >>> loss: ' + str(loss))
            if(epoch%200==0 and epoch!=0):
                saver.save(sess, checkpoint_dir + '/model.ckpt', global_step=epoch)
                # writer.flush()



'''
with tf.Graph().as_default():
    with tf.Session() as sess:

        batchSize = 1
        generator = CaptionGenerator(batchSize, dropout=False)
        _, caption = generator.build_sampler()

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, 'vgg/vgg16.ckpt')

        img = cv2.imread('image/12021.jpeg')
        img = (img.astype(np.float32)/255.0 - 0.5)*2

        inputs = np.zeros((batchSize, 128, 192, 3))
        rots = np.zeros(batchSize, dtype=np.uint8)
        if(img.shape[0]>img.shape[1]):
            img = img.transpose((1,0,2))
            rots[0] = 1
        inputs[0,:,:,:] = img

        feed_dict = {generator.images: inputs, generator.rotates: rots}
        labels = sess.run(caption, feed_dict=feed_dict)
        print("Labels: \n")
        for i in range(len(labels)):
            print(labels[i])
'''

