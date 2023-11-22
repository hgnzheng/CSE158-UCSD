"""
Fall 2023 CSE 158 Midterm Exam, November 7th
Name: Hargen Zheng
PID: A17383701
"""

# %%
import json
import gzip
import math
from collections import defaultdict
import numpy as np
from sklearn import linear_model
import random
import statistics

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
answers = {}

# %%
z = gzip.open("train.json.gz")

# %%
dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)

# %%
z.close()

# %%
### Question 1

# %%
def MSE(y, ypred):
    # Compute the square of each term.
    ds = [(a-b)**2 for (a,b) in zip(y, ypred)]
    # Compute the average sum of squares.
    return sum(ds) / len(ds)

# %%
def MAE(y, ypred):
    # Compute the absolute value of each term.
    ds = [abs(a-b) for (a,b) in zip(y, ypred)]
    # Compute the average absolute values.
    return sum(ds) / len(ds)

# %%
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])

# %%
def feat1(d):
    # Offset term.
    feat = [1]
    # hours as a feature.
    feat.append(d['hours'])
    return feat

# %%
X = [feat1(d) for d in dataset]
y = [len(d['text']) for d in dataset]

# %%
# Fit the linear regression model and make predictions
mod = linear_model.LinearRegression()
mod.fit(X,y)
predictions = mod.predict(X)

# %%
theta_0, theta_1 = mod.coef_ # obtaining coefficients
theta_1

# %%
mse_q1 = MSE(y, predictions) # computing MSE
mse_q1

# %%
answers['Q1'] = [theta_1, mse_q1]
answers['Q1']

# %%
assertFloatList(answers['Q1'], 2)

# %%
### Question 2

# %%
hours = [] # initialize an empty list to store hours across all intersections
for d in dataset:
    hours.append(d['hours']) # add individual hours of intersection

median_hr = statistics.median(hours) # compute median hours of intersection
median_hr

# %%
def feat2(d):
    feat = [1] # offset term
    # compute indicator of whether hours played is above global median.
    hour_indicator = int(d['hours'] > median_hr)
    # Concatenate features together and return the feature vector
    return feat + [d['hours']] + [math.log(d['hours'] + 1, 2)] + \
        [math.sqrt(d['hours'])] + [hour_indicator]

# %%
X = [feat2(d) for d in dataset]

# %%
# Fit the linear regression model and make predictions.
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse_q2 = MSE(y, predictions) # compute MSE value
mse_q2

# %%
answers['Q2'] = mse_q2

# %%
assertFloat(answers['Q2'])

# %%
### Question 3

# %%
def feat3(d):
    # Generate a list of threshold values for easier computation.
    thresholds = [1, 5, 10, 100, 1000]
    feat = [1] # offset term
    # Iterate through the thresholds and append indicator values.
    for threshold in thresholds:
        feat.append(int(d['hours'] > threshold))
    return feat

# %%
X = [feat3(d) for d in dataset]

# %%
# Fit the linear regression model and make predictions.
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse_q3 = MSE(y, predictions) # compute MSE value
mse_q3

# %%
answers['Q3'] = mse_q3

# %%
assertFloat(answers['Q3'])

# %%
### Question 4

# %%
def feat4(d):
    feat = [1] # offset term
    feat.append(len(d['text'])) # review length as a feature
    return feat

# %%
X = [feat4(d) for d in dataset]
y = [d['hours'] for d in dataset]

# %%
# Fit the linear regression model and make predictions.
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse = MSE(y, predictions) # compute MSE value
mae = MAE(y, predictions) # compute MAE value
mse, mae

# %%
explain = "MAE is better suited for this dataset because it seems we might have large outlier(s) in our dataset, which is punished heavily by the MSE metric. Since the average distance is at a much smaller scale than the average squared distance, MAE would be more suitable for this dataset."

# %%
answers['Q4'] = [mse, mae, explain]
answers['Q4']

# %%
assertFloatList(answers['Q4'][:2], 2)

# %%
### Question 5

# %%
def y_transform(d):
    # compute transformation according to the given formula, with base 2.
    return math.log(d['hours'] + 1, 2)

# %%
y_trans = [y_transform(d) for d in dataset]

# %%
# Fit the linear regression model and make predictions.
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)

# %%
mse_trans = MSE(y_trans, predictions_trans) # MSE using the transformed variable
mse_trans

# %%
def pred_untransform(predictions):
    return (2**predictions - 1) # transform back to variable x with math manipulation

# %%
predictions_untrans = pred_untransform(predictions_trans) # Undoing the transformation
predictions_untrans[:10]

# %%
mse_untrans = MSE(y, predictions_untrans) # compute MSE value
mse_untrans

# %%
answers['Q5'] = [mse_trans, mse_untrans]
answers['Q5']

# %%
assertFloatList(answers['Q5'], 2)

# %%
### Question 6

# %%
def feat6(d):
    encode_lst = np.arange(1, 100) # generate list of hours to be compared
    one_hot_lst = np.zeros(100, dtype='int32') # one-hot encoding list
    hr_played = d['hours'] # extract hours played
    if hr_played > 99: # case where hours played is greater than 99
        one_hot_lst[-1] = 1
    else:
        for index in encode_lst: # compare with other number of hours
            if hr_played < index:
                one_hot_lst[index-1] = 1
                break # break out of the loop once one criterion is met
    return [1] + list(one_hot_lst) # concatenate offset and one-hot encoding list

# %%
X = [feat6(d) for d in dataset]
y = [len(d['text']) for d in dataset]

# %%
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

# %%
models = {}
mses = {}
bestC = None
bestValidMSE = None

for c in [1, 10, 100, 1000, 10000]:
    # Fit the linear regression model with training data and make predictions for a given alpha.
    mod = linear_model.Ridge(alpha=c)
    mod.fit(Xtrain, ytrain)
    models[c] = mod
    # Generate predictions for validation and test datasets.
    y_pred_valid = mod.predict(Xvalid)
    y_pred_test = mod.predict(Xtest)
    # Compute MSE values accordingly.
    mse_valid = MSE(yvalid, y_pred_valid)
    mse_test = MSE(ytest, y_pred_test)
    # Add MSE value into the dictionary
    mses[c] = [mse_valid, mse_test]
    # Do comparison to update best value of C and best validation MSE value.
    if bestC == None or mse_valid < bestValidMSE:
        bestC = c
        bestValidMSE = mse_valid

# %%
bestC

# %%
mse_valid = mses[bestC][0]

# %%
mse_test = mses[bestC][1]

# %%
answers['Q6'] = [bestC, mse_valid, mse_test]
answers['Q6']

# %%
assertFloatList(answers['Q6'], 3)

# %%
### Question 7

# %%
times = [d['hours_transformed'] for d in dataset] # extract hours_transformed for each data
median = statistics.median(times) # compute the median for hours_transformed
median

# %%
def play_indicator(d):
    # Indicate True if hours played is less than 1
    if d['hours'] < 1:
        return 1
    # Indicate False otherwise
    return 0

# %%
notPlayed = [play_indicator(d) for d in dataset] # generate a list of indicators
nNotPlayed = sum(notPlayed) # count the intersections with less than 1 hour play time
nNotPlayed

# %%
answers['Q7'] = [median, nNotPlayed]
answers['Q7']

# %%
assertFloatList(answers['Q7'], 2)

# %%
### Question 8

# %%
def feat8(d):
    feat = [1] # offset term
    feat.append(len(d['text'])) # review length as a feature
    return feat

# %%
X = [feat8(d) for d in dataset]
y = [d['hours_transformed'] > median for d in dataset]

# %%
# Fit the linear regression model and make predictions.
mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions

# %%
# Function to obtain evaluation metrics
def rates(predictions, y):
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    
    return TP, TN, FP, FN

# %%
TP, TN, FP, FN = rates(predictions, y) # compute evaluation metrics

# %%
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)

BER = 1 - 1/2 * (TPR + TNR) # Calculate BER

# %%
answers['Q8'] = [TP, TN, FP, FN, BER]
answers['Q8']

# %%
assertFloatList(answers['Q8'], 5)

# %%
### Question 9

# %%
scores = mod.decision_function(X) # calculate probability scores

# %%
scorelabels = list(zip(scores, y))
scorelabels.sort(reverse=True)
scorelabels[:10] # generate a sorted list of score and label pairs

# %%
sortedlabels = [x[1] for x in scorelabels] # generate a list of labels

# %%
precs = []
recs = []

for i in [5, 10, 100, 1000]:
    score_k = scorelabels[i][0] # obtain the probability score@k
    while scorelabels[i][0] == score_k:
            i+=1 # increment k until tie is broken
    precs.append(sum(sortedlabels[:i]) / i) # compute precision@k after tie breaking

# %%
answers['Q9'] = precs
answers['Q9']

# %%
assertFloatList(answers['Q9'], 4)

# %%
### Question 10

# %%
y_trans = [d['hours_transformed'] for d in dataset]

# %%
# Fit the linear regression model and make predictions.
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)

# %%
# Obtain the interval of which predictions_trans take values.
min, max = int(np.min(predictions_trans)), int(np.max(predictions_trans))
min, max

# %%
q8_BER = answers['Q8'][4]
q8_BER

# %%
predictions_thresh = min # Using a fixed threshold to make predictions
bestBER = None

# y contains True/False labels if the transformed hours is greater than median.

# for i in np.arange(np.min(predictions_trans), np.max(predictions_trans), 0.05):
for i in np.arange(min, max, 0.05):
    # Compute labels with the given threshold
    label_pred = [x > i for x in predictions_trans]
    
    # Compute BER values according to the given threshold
    TP, TN, FP, FN = rates(label_pred, y)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    BER = 1 - 1/2 * (TPR + TNR)
    
    # Compare with q8 BER value, update variables accordingly.
    if BER < q8_BER:
        bestBER = BER
        predictions_thresh = i
        break # break out as soon as we find a better BER value

# %%
bestBER, predictions_thresh

# %%
answers['Q10'] = [predictions_thresh, bestBER]
answers['Q10']

# %%
assertFloatList(answers['Q10'], 2)

# %%
### Question 11

# %%
dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]

# %%
userMedian = defaultdict(list)
itemMedian = defaultdict(list)

# Compute medians on training data
for d in dataTrain:
    # Add user into the userMedian if not yet 
    if d['userID'] not in userMedian:
        userMedian[d['userID']] = [d['hours']]
    # Append user play hours if user is already in the dictionary
    else:
        userMedian[d['userID']] += [d['hours']]
    # Add item into the itemMedian if not yet 
    if d['gameID'] not in itemMedian:
        itemMedian[d['gameID']] = [d['hours']]
    # Append item play hours if user is already in the dictionary
    else:
        itemMedian[d['gameID']] += [d['hours']]

# %%
userMedian = {d: statistics.median(userMedian[d]) for d in userMedian} # compute median play hours for each user
itemMedian = {d: statistics.median(itemMedian[d]) for d in itemMedian} # compute median play hours for each game

# %%
answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]
answers['Q11']

# %%
assertFloatList(answers['Q11'], 2)

# %%
### Question 12

# %%
def f12(u,i):
    # Function returns a single value (0 or 1), global median stored in median_hr
    # Case where item is present and its median play hours is greater than global median
    if i in itemMedian and itemMedian[i] > median_hr:
        return 1
    # Case where item is not present but user's median play hours is greater than global median
    if i not in itemMedian and userMedian[u] > median_hr:
        return 1
    # 0 otherwise
    return 0

# %%
preds = [f12(d['userID'], d['gameID']) for d in dataTest]

# %%
y = [1 if d['hours'] > median_hr else 0 for d in dataTest] # label by comparing to global median

# %%
accuracy = sum([a==b for a,b in zip(preds,y)]) / len(y) # compute accuracy

# %%
answers['Q12'] = accuracy
answers['Q12']

# %%
assertFloat(answers['Q12'])

# %%
### Question 13

# %%
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}

for d in dataset:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)

# %%
def Jaccard(s1, s2):
    # Size of common items for both s1 and s2
    numer = len(s1.intersection(s2))
    # Size of all items in either s1 or s2
    denom = len(s1.union(s2))
    # Edge case
    if denom == 0:
        return 0
    return numer / denom

# %%
def mostSimilar(i, func, N):
    # Initialize a list to store similarities.
    similarities = []
    # Get users for the given item.
    users = usersPerItem[i]
    # Iterate through items.
    for i2 in usersPerItem:
        # Discard item that is the same as the input item.
        if i2 == i: continue
        # Compute similarity with the input function.
        sim = func(users, usersPerItem[i2])
        # Append similarity to the similarity list.
        similarities.append((sim,i2))
    # Sort similarities.
    similarities.sort(reverse=True)
    # Return N most similar items.
    return similarities[:N]

# %%
ms = mostSimilar(dataset[0]['gameID'], Jaccard, 10) # obtain 10 most similar games

# %%
answers['Q13'] = [ms[0][0], ms[-1][0]]
answers['Q13']

# %%
assertFloatList(answers['Q13'], 2)

# %%
### Question 14

# %%
def mostSimilar14(i, func, N):
    # Initialize a list to store similarities
    similarities = []
    # Obtain a list of users for the item.
    users = usersPerItem[i]
    # Iterate through list of items.
    for i2 in usersPerItem:
        # Discard item that is the same as the input item.
        if i2 == i: continue
        # Compute similarity between two items with the input function.
        sim = func(i, i2)
        # Append similarity to the list.
        similarities.append((sim,i2))
    # Sort similarities.
    similarities.sort(reverse=True)
    # Return N most similar items.
    return similarities[:N]

# %%
ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = 1 if d['hours'] > median_hr else -1 # Set the label based on a rule
    ratingDict[(u,i)] = lab

# %%
def Cosine(i1, i2):
    # Between two items
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = 0
    denom1 = 0
    denom2 = 0
    # Computing cosine similarity metrics
    for u in inter:
        numer += ratingDict[(u,i1)]*ratingDict[(u,i2)]
    for u in usersPerItem[i1]:
        denom1 += ratingDict[(u,i1)]**2
    for u in usersPerItem[i2]:
        denom2 += ratingDict[(u,i2)]**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom

# %%
ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)

# %%
answers['Q14'] = [ms[0][0], ms[-1][0]]
answers['Q14']

# %%
assertFloatList(answers['Q14'], 2)

# %%
### Question 15

# %%
ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = d['hours_transformed'] # Set the label based on a rule
    ratingDict[(u,i)] = lab

# %%
ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)

# %%
answers['Q15'] = [ms[0][0], ms[-1][0]]
answers['Q15']

# %%
assertFloatList(answers['Q15'], 2)

# %%
f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



