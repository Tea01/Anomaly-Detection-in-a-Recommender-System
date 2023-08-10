# -*- coding: utf-8 -*-
"""
# Data

The dataset consists of a rating matrix used for sparse representation in a recommender system. The numpy arrays _train_1.npy_, _train_2.npy_, _test_1.npy_, and _test_2.npy_ contain information on user and item IDs, as well as their corresponding ratings. Each line in these arrays represents an entry in the rating matrix.

In the following examples, you can see the format of the file train_1.npy. Each row in the numpy array contains information about a user-item interaction. The first column represents the user ID, the second column represents the item ID, and the third column represents the corresponding rating given by the user to that item. For example, in the first row of the numpy array, we can see that user 1 has given a rating of 4 stars to item 1382.
"""

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
np.load("ratings_supervised.npy")

"""In the file _train_1_labels.npy_, the labels are pairs of user IDs with their respective labels. A label of 1 indicates an anomaly in the user's behavior, while a label of 0 indicates a normal user. However, labels are not available for the sets _train_2.npy_, _test_1.npy_, and _test_2.npy_.

"""

!pip install implicit

print(np.min(user_ids))

import numpy as np

train_data = np.load('ratings_supervised.npy')
train_labels = np.load('ratings_supervised_label.npy')
test_data = np.load('ratings_leaderboard.npy')
data = np.concatenate((train_data, test_data), axis=0)

# Create a sparse rating matrix from the training data
user_ids = data[:, 0].astype(int) - 45001
item_ids = data[:, 1].astype(int) - 2250
ratings = data[:, 2].astype(float)
num_users = np.max(user_ids) + 1
num_items = np.max(item_ids) + 1
rating_matrix = csr_matrix((ratings, (user_ids, item_ids)), shape=(num_users, num_items))
print(num_users)
print(num_items)

num_of_ratings = np.empty((num_users,1))
num_of_fives = np.empty((num_users, 1))
num_of_four = np.empty((num_users, 1))
num_of_three = np.empty((num_users, 1))
num_of_two = np.empty((num_users, 1))
num_of_one = np.empty((num_users, 1))
average_rating = np.empty((num_users,1))
for i in range(num_users):
  num_of_ratings[i,0] = np.nonzero(rating_matrix[i,:])[1].shape[0]
  num_of_fives[i,0] = np.nonzero(rating_matrix[i,:] == 5)[1].shape[0]
  num_of_four[i,0] = np.nonzero(rating_matrix[i,:] == 4)[1].shape[0]
  num_of_three[i,0] = np.nonzero(rating_matrix[i,:] == 3)[1].shape[0]
  num_of_two[i,0] = np.nonzero(rating_matrix[i,:] == 2)[1].shape[0]
  num_of_one[i,0] = np.nonzero(rating_matrix[i,:] == 1)[1].shape[0]
  average_rating[i,0] = np.mean(rating_matrix[i,:][rating_matrix[i,:]!=0])

C = np.concatenate((num_of_ratings, num_of_fives,num_of_four,num_of_three,num_of_two,num_of_one,average_rating), axis=1)

print(C)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix

#train_data = np.load('train_1.npy')
#train_labels = np.load('train_1_labels.npy')
#test_data = np.load('test_1.npy')
#data = np.concatenate((train_data, test_data), axis=0)

train_data = np.load('ratings_supervised.npy')
train_labels = np.load('ratings_supervised_label.npy')
test_data = np.load('ratings_leaderboard.npy')
data = np.concatenate((train_data, test_data), axis=0)

# Create a sparse rating matrix from the training data
user_ids = data[:, 0].astype(int) - 45001
item_ids = data[:, 1].astype(int) - 2250
ratings = data[:, 2].astype(float)
num_users = np.max(user_ids) + 1
num_items = np.max(item_ids) + 1
rating_matrix = csr_matrix((ratings, (user_ids, item_ids)), shape=(num_users, num_items))


# Train the matrix factorization model
model = AlternatingLeastSquares(factors=50, regularization=0.01)
model.fit(rating_matrix)

# Train a logistic regression model
clf = LogisticRegression(random_state=0, max_iter=1000, solver='liblinear')
param = {'penalty':['l1', 'l2'], 'C':10**np.linspace(-3, 3, 20, endpoint=True)}
lr_gs = GridSearchCV(clf, param, n_jobs=-1, cv=10)
U_train = model.user_factors.to_numpy()[:5000]
V_train = model.item_factors.to_numpy()
R_train = U_train @ V_train.T

featuresU = np.concatenate((U_train, C[:5000]), axis = 1)
featuresR = np.concatenate((R_train, C[:5000]), axis = 1)

# We should try featuresU and R, those include more information about each user
lr_gs.fit(featuresU, train_labels[:, 1]);

# Calculate the predictions using threshold
U_test = model.user_factors.to_numpy()[5000:]
R_test = U_test @ V_train.T

featuresUtest = np.concatenate((U_test, C[5000:]), axis = 1)

test_pred = lr_gs.predict_proba(featuresUtest)
test_thresh = test_pred[:, 0] < 0.8
print('Non-zero elements: ', np.count_nonzero(test_thresh))
int_pred = test_thresh.astype(int)
str_pred = ''.join(str(x) for x in int_pred)
np.savetxt('leaderboard_labels.txt', [str_pred], delimiter='', fmt='%s')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix

#train_data = np.load('train_1.npy')
#train_labels = np.load('train_1_labels.npy')
#test_data = np.load('test_1.npy')
#data = np.concatenate((train_data, test_data), axis=0)

train_data = np.load('ratings_supervised.npy')
train_labels = np.load('ratings_supervised_label.npy')
test_data = np.load('ratings_leaderboard.npy')
data = np.concatenate((train_data, test_data), axis=0)

# Create a sparse rating matrix from the training data
user_ids = data[:, 0].astype(int) - 45001
item_ids = data[:, 1].astype(int) - 2250
ratings = data[:, 2].astype(float)
num_users = np.max(user_ids) + 1
num_items = np.max(item_ids) + 1
rating_matrix = csr_matrix((ratings, (user_ids, item_ids)), shape=(num_users, num_items))


# Train the matrix factorization model
model = AlternatingLeastSquares(factors=50, regularization=0.01)
model.fit(rating_matrix)

U_train = model.user_factors.to_numpy()[:5000]
V_train = model.item_factors.to_numpy()

featuresU = np.concatenate((U_train, C[:5000]), axis = 1)

# Train a logistic regression model
param = {'n_estimators': [5, 10, 25, 50, 100],
         'criterion':['gini','entropy'],
         'max_depth':[None, 2, 3, 4, 5],
         'min_samples_leaf':[1, 2, 5, 10]}

rf =  RandomForestClassifier(random_state=0)
rf_bs = GridSearchCV(rf, param, cv= 5)
rf_bs.fit(featuresU, train_labels[:, 1]);

# Calculate the predictions using threshold
U_test = model.user_factors.to_numpy()[5000:]
R_test = U_test @ V_train.T

featuresUtest = np.concatenate((U_test, C[5000:]), axis = 1)

test_pred = rf_bs.predict_proba(featuresUtest)
test_thresh = test_pred[:, 0] < 0.85
print('Non-zero elements: ', np.count_nonzero(test_thresh))
int_pred = test_thresh.astype(int)
str_pred = ''.join(str(x) for x in int_pred)
np.savetxt('leaderboard_labels.txt', [str_pred], delimiter='', fmt='%s')

print(rf_bs.best_params_)