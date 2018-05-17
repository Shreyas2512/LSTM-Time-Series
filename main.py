import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import data_loader
import matplotlib.pyplot as plt

class SeriesPredictor:

	def __init__(self, input_dim, seq_size, hidden_dim):
		## This function is invoked when the object of this class is created
		## Hyperparameters are initialized
		self.input_dim = input_dim
		self.seq_size = seq_size
		self.hidden_dim = hidden_dim

		##Weight variables and input placeholders
		##The output is taken across all the time steps as each time step predicts the output of time series at next time step
		##So output is num_examples,seq_length		
		self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
		self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
		self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
		self.y = tf.placeholder(tf.float32, [None, seq_size])

		##The cost function here is least square error of output of each time step with actual data
		#self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
		self.cost = tf.reduce_mean(tf.abs(self.model() - self.y))		
		self.train_op = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(self.cost)

		# Auxiliary ops
		self.saver = tf.train.Saver()

	def model(self):
		
		cell = rnn.BasicLSTMCell(self.hidden_dim)
		##Shape of outputs is number_examples, 5, 100		
		outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
		num_examples = tf.shape(self.x)[0]
		W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
		out = tf.matmul(outputs, W_repeated) + self.b_out
		##Shape of out is (?, 5, 1)		
		out = tf.squeeze(out)
		##After squeezing the predicted output shape of model has changed to num_examples,seq_length which is same shape as that of actual output
		return out

	def train(self, train_x, train_y, test_x, test_y):
		with tf.Session() as sess:
			tf.get_variable_scope().reuse_variables()
			sess.run(tf.global_variables_initializer())
			self.saver.restore(sess, './model.ckpt')			
			##The patience decreases as the train loss does not decrease further			
			max_patience = 30000
			patience = max_patience
			min_test_err = float('inf')
			step = 0
			while patience > 0:
				_, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
				if step % 100 == 0:
					self.saver.save(sess, '/home/ashish/time_series_analysis/loss_insertion/model.ckpt')
					test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
					print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
					if test_err < min_test_err:
						min_test_err = test_err
						patience = max_patience
					else:
						patience -= 1
				step += 1
			save_path = self.saver.save(sess, '/home/ashish/time_series_analysis/loss_insertion/model.ckpt')
			print('Model saved to {}'.format(save_path))

	def test(self, sess, test_x):
		tf.get_variable_scope().reuse_variables()
		self.saver.restore(sess, './model.ckpt')
		output = sess.run(self.model(), feed_dict={self.x: test_x})
		return output


def plot_results(train_x, predictions, actual, filename):
	plt.figure()
	num_train = len(train_x)
	plt.plot(list(range(num_train)), train_x, color='b', label='training data')
	plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
	plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
	plt.legend()
	#if filename is not None:
	#	plt.savefig(filename)
	#else:
	plt.show()

def plot_results_predict(train_x, predictions, valid, test, filename):
	plt.figure()
	num_train = len(train_x)
	plt.plot(list(range(num_train)), train_x, color='b', label='training data')
	plt.plot(list(range(num_train + len(valid), num_train + len(valid) + len(test))), test, color='y', label='predicted')
	plt.plot(list(range(num_train, num_train + len(valid))), valid, color='g', label='test data')
	plt.plot(list(range(num_train + len(valid), num_train + len(valid) + len(predictions))), predictions, color='r', label='predicted')
	plt.legend()
	#if filename is not None:
	#	plt.savefig(filename)
	#else:
	plt.show()


if __name__ == '__main__':
	seq_size = 20
	##Performing all the data operations	
	predictor = SeriesPredictor(input_dim = 1, seq_size = seq_size, hidden_dim = 100)
	data = data_loader.load_series('datanew.csv')
	train_data, valid_data, test_data = data_loader.split_data(data)

	'''
	print("How Train data looks like")
	for i in range(10):
		print(train_data[i])
	'''
	

	##Here we are making the data s.t the output at every time step is the value for the next time-step
	train_x, train_y = [], []
	for i in range(len(train_data) - seq_size - 1):
		##Expand_dims is used since we have to feed the network with an input that has first dimension as batch_size, second dimension as seq_length and third
		##dimension as input_dim
		##The first dimension is fulfilled by appending many lists to train_x, third dimension is fulfilled by using expand_dims		
		train_x.append(np.expand_dims(train_data[i : i + seq_size], axis = 1).tolist())
		train_y.append(train_data[i+1 : i + seq_size + 1])

	'''
	print("How input train data to the network looks like")
	for i in range(10):
		print(train_x[i])

	print("How output train data to the network looks like")
	for i in range(10):
		print(train_y[i])
	'''


	valid_x, valid_y = [], []
	for i in range(len(valid_data) - seq_size - 1):
		valid_x.append(np.expand_dims(valid_data[i : i + seq_size], axis = 1).tolist())
		valid_y.append(valid_data[i + 1 : i + seq_size + 1])

	test_x, test_y = [], []
	for i in range(len(test_data) - seq_size - 1):
		test_x.append(np.expand_dims(test_data[i : i + seq_size], axis = 1).tolist())
		test_y.append(test_data[i + 1 : i + seq_size + 1])


	
	#plot_results(train_data, valid_data, test_data, 'predictions.png')
	'''
	print("How input test data to the network looks like")
	for i in range(10):
		print(test_x[i])

	print("How output test data to the network looks like")
	for i in range(10):
		print(test_y[i])
	'''

	predictor.train(train_x, train_y, valid_x, valid_y)

	'''
	with tf.Session() as sess:
		predicted_vals = predictor.test(sess, valid_x)[:, 0]
		##Number of examples in predicted_vals is same as that of test_x 		
		print('predicted_vals', np.shape(predicted_vals))
		plot_results(train_data, predicted_vals, valid_data, 'predictions.png')
		prev_seq = valid_x[-1]
		predicted_vals = []
		for i in range(40):
			next_seq = predictor.test(sess, [prev_seq])
			predicted_vals.append(next_seq[-1])
			prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
			plot_results_predict(train_data, predicted_vals, valid_data, test_data, 'hallosinations.png')
	'''
