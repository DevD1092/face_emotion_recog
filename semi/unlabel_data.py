import pandas
import numpy as np

#load dataset
data = pandas.read_csv("fer2013.csv", delimiter=",")
data = data.values

#Number of training data per class in the original dataset
# 0 - 3995
# 1 - 436
# 2 - 4097
# 3 - 7215
# 4 - 4830
# 5 - 3171
# 6 - 4965

X_label = []
Y_label = []
X_unlabel = []
Y_unlabel = []

#Segregating the dataset
train_X = data[0:28708,1].astype(str)
train_Y = data[0:28708,0].astype(int)

#print train_X[0]
#print train_Y[0]
#Counting variables
c_0, c_1, c_2, c_3, c_4, c_5, c_6 = 0, 0, 0, 0, 0, 0, 0

# Segregating the dataset - Unlabeled and labeled
for id, val in enumerate(train_Y):
	if val == 0:
		if c_0 < 799:
			X_label.append(train_X[id])
			Y_label.append(train_Y[id])
			c_0 = c_0 + 1
		else:
			X_unlabel.append(train_X[id])
			Y_unlabel.append(train_Y[id])
	if val == 1:
		if c_1 < 87:
			X_label.append(train_X[id])
			Y_label.append(train_Y[id])
			c_1 = c_1 + 1
		else:
			X_unlabel.append(train_X[id])
			Y_unlabel.append(train_Y[id])
	if val == 2:
		if c_2 < 818:
			X_label.append(train_X[id])
			Y_label.append(train_Y[id])
			c_2 = c_2 + 1
		else:
			X_unlabel.append(train_X[id])
			Y_unlabel.append(train_Y[id])
	if val == 3:
		if c_3 < 1442:
			X_label.append(train_X[id])
			Y_label.append(train_Y[id])
			c_3 = c_3 + 1
		else:
			X_unlabel.append(train_X[id])
			Y_unlabel.append(train_Y[id])
	if val == 4:
		if c_4 < 966:
			X_label.append(train_X[id])
			Y_label.append(train_Y[id])
			c_4 = c_4 + 1
		else:
			X_unlabel.append(train_X[id])
			Y_unlabel.append(train_Y[id])
	if val == 5:
		if c_5 < 634:
			X_label.append(train_X[id])
			Y_label.append(train_Y[id])
			c_5 = c_5 + 1
		else:
			X_unlabel.append(train_X[id])
			Y_unlabel.append(train_Y[id])
	if val == 6:
		if c_6 < 992:
			X_label.append(train_X[id])
			Y_label.append(train_Y[id])
			c_6 = c_6 + 1
		else:
			X_unlabel.append(train_X[id])
			Y_unlabel.append(train_Y[id])

np.savetxt("X_label_20.csv", X_label, delimiter=" ", fmt = '%s')
np.savetxt("Y_label_20.csv", Y_label, delimiter=" ")
np.savetxt("X_unlabel_80.csv", X_unlabel, delimiter=" ", fmt = '%s')
np.savetxt("Y_unlabel_80.csv", Y_unlabel, delimiter=" ")

#with open('myfile.csv', 'w') as myfile:
#	wr = csv.writer(myfile,quoting=csv.QUOTE_ALL)
#	wr.writerow(X_label)

