import tensorflow as tf
import numpy as np
from copy import copy

# parameters to control terminal conditions
nMaximumEpochs = 1E7 #truned off, recommend 80~100
maxPatience = 20 	#in steps
usePatience = True
nBatchSize = 128 #currently this script requires a batchsize of 128


def predict(x, w01, w12, b01, b12):
	result1 = tf.nn.tanh(tf.matmul(x, w01) + b01)
	result2 = tf.nn.relu(tf.matmul(result1, w12) + b12)
	return result2

def optimizeModel(dataSets, nHidden , initWeights):

	(trainX, testX, trainY, testY) = dataSets
	nTarget = trainY.shape[1]
	nSamples = trainX.shape[0]
	nInput = trainX.shape[1]
	nStepsEpoch = np.ceil(nSamples/nBatchSize)
	maximumSteps = nStepsEpoch * nMaximumEpochs

	tf.reset_default_graph()
	w01, w12 = initWeights
	weights01 = tf.Variable(w01)
	weights12 = tf.Variable(w12)
	biases01 = tf.Variable(np.zeros([nHidden]).astype(np.float32))
	biases12 = tf.Variable(np.zeros([nTarget]).astype(np.float32))
	xPlaceholder = tf.placeholder(tf.float32, [nBatchSize, nInput])
	yPlaceholder = tf.placeholder(tf.float32, [nBatchSize, nTarget])
		
	prediction = predict(xPlaceholder,weights01,weights12,biases01,biases12)
	L2norm = tf.reduce_mean(tf.pow(prediction - yPlaceholder,2))
	optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(L2norm)

	initOperation = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(initOperation)
		nSteps = 1
		patience = copy(maxPatience)
		bestAccuracy =  1E6
		currentAccuracy = 1E6
		while nSteps < maximumSteps and patience > 0:
			ind = nSteps % (nSamples-nBatchSize)
			batchX = trainX[ind : ind + nBatchSize,:]
			batchY = trainY[ind : ind + nBatchSize, :]
			sess.run(optimizer, feed_dict = {xPlaceholder: batchX, yPlaceholder: batchY})
			nSteps += 1
			if 0 == nSteps % nStepsEpoch:
				previousAccuracy = copy(currentAccuracy)
				currentAccuracy = sess.run(L2norm, feed_dict={xPlaceholder: testX, yPlaceholder: testY})

				patience *= 0.9
				if currentAccuracy < previousAccuracy:
					patience += 0.8
				else:
					patience -= 1
				if currentAccuracy < bestAccuracy:
					bestAccuracy = currentAccuracy

		npW01 = sess.run(weights01, feed_dict = {xPlaceholder: batchX, yPlaceholder: batchY})
		npW12 = sess.run(weights12, feed_dict={xPlaceholder: batchX, yPlaceholder: batchY})
		npB01 = sess.run(biases01, feed_dict={xPlaceholder: batchX, yPlaceholder: batchY})
		npB12 = sess.run(biases12, feed_dict={xPlaceholder: batchX, yPlaceholder: batchY})

	return (bestAccuracy,npW01, npW12, npB01,npB12)
	   
