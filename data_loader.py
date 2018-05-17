import csv
import numpy as np
import matplotlib.pyplot as plt


def load_series(filename, series_idx=0):
	try:
		with open(filename) as csvfile:
			csvreader = csv.reader(csvfile)

			data = [float(row[series_idx]) for row in csvreader if len(row) > 0]
			#normalized_data = (data - np.mean(data)) / np.std(data)
		return data[0:1179]
	except IOError:
		return None

def split_data(data, percent_train=0.80):
	num_rows = len(data)
	train_amount = percent_train*num_rows
	rem_amount = num_rows - train_amount
	valid_amount = 0.5*rem_amount
	test_amount = 0.5*rem_amount
	train_data, valid_data, test_data = [], [], []
	for idx, row in enumerate(data):
		if idx < train_amount:
			train_data.append(row)
		else:
			if idx < train_amount + valid_amount:
				valid_data.append(row)
			else:
				test_data.append(row)
	return train_data, valid_data, test_data


timeseries = load_series('datanew.csv')
#print(np.shape(timeseries))

plt.figure()
plt.plot(timeseries)
plt.show()
