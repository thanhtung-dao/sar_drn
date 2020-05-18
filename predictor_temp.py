import argparse
import numpy as np
import pickle
import os
from cv2 import cv2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from model import get_red30_model,PSNR ############

model = get_red30_model()
model.load_weights('./checkpoints/weights.020.hdf5')

img_name = 'buildings37'
img = cv2.imread('./images/' + img_name + '.png')
img = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2])
img_pred = model.predict(img)
img_pred = img_pred.reshape(img_pred.shape[1], img_pred.shape[2], img_pred.shape[3])
cv2.imwrite('./images/' + img_name + '_pred.png', img_pred)
print('---COMPLETE---')