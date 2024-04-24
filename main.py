import os
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
from model import Recommender
import random
if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')
	np.random.seed(100)
	random.seed(100)
	tf.set_random_seed(100)
	with tf.Session(config=config) as sess:
		recom = Recommender(sess, handler)
		recom.run()