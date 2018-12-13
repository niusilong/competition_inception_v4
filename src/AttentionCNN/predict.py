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


with tf.Graph().as_default():
    with tf.Session() as sess:

        batchSize = 1
        generator = CaptionGenerator(batchSize, dropout=False)
        _, caption = generator.build_sampler()

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, 'step/model.ckpt-200')

	files = os.listdir('3')
	intFiles = []
	for file in files:
	    intFiles.append(int(file.split('.')[0]))
	intFiles.sort()
	resultFile = open('result.txt','w')

	for f in intFiles:
	    	#print(f)

            img = cv2.imread('3/'+str(f)+'.jpeg')
            img = cv2.resize(img, (299, 299))
            #img = (img.astype(np.float32)/255.0 - 0.5)*2

            inputs = np.zeros((1, 299, 299, 3))
            inputs[0,:,:,:] = img

            feed_dict = {generator.images: inputs}
            labels = sess.run(caption, feed_dict=feed_dict)
            labels = np.unique(labels[0])
            #print("Labels: "+str(len(labels))+'\n')
            for i,lb in enumerate(labels):
                if(int(lb)==261):
                    resultFile.write('\r\n')
                    break
                elif(i!=0):
                    resultFile.write(',')
                elif(i==0):
                    resultFile.write(str(f)+' ')
                resultFile.write(str(lb))
