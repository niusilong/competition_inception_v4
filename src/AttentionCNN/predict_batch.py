#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import tensorflow as tf
import cv2
from model import CaptionGenerator
import random
import re
from PIL import Image, ImageEnhance
import threading


imgDir = '3/'

with tf.Session() as sess:

    batchSize = 30
    generator = CaptionGenerator(batchSize, dropout=False)
    _, caption = generator.build_sampler()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'step/model.ckpt-1200')

    files = os.listdir(imgDir)
    intFiles = []
    for file in files:
        intFiles.append(int(file.split('.')[0]))
    intFiles.sort()
    resultFile = open('result.txt','w')


    batchs = []
    rimages = []
    for i,f in enumerate(intFiles):

        img = cv2.imread(imgDir+str(f)+'.jpeg')[...,::-1]
        img = cv2.resize(img, (299, 299))
        # img = (img.astype(np.float32)/255.0 - 0.5)*2

        # rimages = []
        rimages.append(img)

        if((i+1)%30==0):
            batchs.append(rimages)
            rimages = []


    for i, batch in enumerate(batchs):
        feed_dict = {generator.images: batch}
        labels = sess.run(caption, feed_dict=feed_dict)

        for j,lb in enumerate(labels):
            lb = np.unique(lb)
            for k in range(7):

                if(int(lb[k])==261):
                    resultFile.write('\r\n')
                    break
                elif(k!=0):
                    resultFile.write(',')
                elif(k==0):
                    resultFile.write(str(intFiles[i*30+j])+' ')
                resultFile.write(str(lb[k]))


