"""
CSE 158/258 Homework 1
Name: Hargen Zheng
PID: A17383701
Source: consulted week1/2 lecture slides and 'Chapter 3 Workbook'
"""
# %%
import json
from collections import defaultdict
from sklearn import linear_model
import numpy
import random
import gzip
import dateutil.parser
import math

# %%
answers = {}

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
### Question 1

# %%
f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

# %%
dataset[0]

# %%
max_len = max([len(d['review_text']) for d in dataset])
max_len

# %%
def feature(datum):
    a = len(datum['review_text']) / max_len
    return [1, a]

# %%
X = [feature(data) for data in dataset]
Y = [d['rating'] for d in dataset]

# %%
model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X, Y)
theta = model.coef_
theta

# %%
# MSE and R^2
y_pred = model.predict(X)

# Sum of squared errors (SSE)
sse = sum([x ** 2 for x in (Y - y_pred)])

MSE = sse / len(Y)
MSE

# %%
answers['Q1'] = [theta[0], theta[1], MSE]

# %%
assertFloatList(answers['Q1'], 3)

# %%
### Question 2

# %%
for d in dataset:
    t = dateutil.parser.parse(d['date_added'])
    d['parsed_date'] = t


# %%
lst = [d['parsed_date'] for d in dataset]
weekdays = [date.weekday() for date in lst]
months = [date.month for date in lst]
print(f"weekdays interval is {max(weekdays) - min(weekdays)}")
print(f"month interval is {max(months) - min(months)}")

# %%
max_len = max([len(d['review_text']) for d in dataset])
max_len

# %%
def feature(datum):
    length = len(datum['review_text']) / max_len
    
    weekday = datum['parsed_date'].weekday()
    month = datum['parsed_date'].month
    weekday_encode = [0 for x in range(6)]
    month_encode = [0 for x in range(11)]
    
    d_idx = weekday - 1
    if d_idx >= 0:
        weekday_encode[d_idx] = 1
    
    m_idx = month - 2
    if m_idx >= 0:
        month_encode[m_idx] = 1
    return [1, length] + weekday_encode + month_encode

# %%
X2 = [feature(d) for d in dataset]
Y2 = [d['rating'] for d in dataset]

# %%
model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X2, Y2)
theta = model.coef_
theta

# %%
# MSE and R^2
y_pred = model.predict(X2)

# Sum of squared errors (SSE)
sse = sum([x ** 2 for x in (Y2 - y_pred)])

mse2 = sse / len(Y)
mse2

# %%
answers['Q2'] = [X2[0], X2[1]]

# %%
assertFloatList(answers['Q2'][0], 19)
assertFloatList(answers['Q2'][1], 19)

# %%
### Question 3

# %%
def feature(datum):
    length = len(datum['review_text']) / max_len
    
    weekday = datum['parsed_date'].weekday()
    month = datum['parsed_date'].month
    return [1, length, weekday, month]

# %%
X3 = [feature(d) for d in dataset]
Y3 = [d['rating'] for d in dataset]

# %%
X3[:10]

# %%
model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X3, Y3)
theta = model.coef_
theta

# %%
# MSE and R^2
y_pred = model.predict(X3)

# Sum of squared errors (SSE)
sse = sum([x ** 2 for x in (Y3 - y_pred)])

mse3 = sse / len(Y3)
mse3

# %%
answers['Q3'] = [mse2, mse3]

# %%
assertFloatList(answers['Q3'], 2)

# %%
### Question 4

# %%
random.seed(0)
random.shuffle(dataset)

# %%
X2 = [feature(d) for d in dataset]
X3 = [feature(d) for d in dataset]
Y = [d['rating'] for d in dataset]

# %%
train2, test2 = X2[:len(X2)//2], X2[len(X2)//2:]
train3, test3 = X3[:len(X3)//2], X3[len(X3)//2:]
trainY, testY = Y[:len(Y)//2], Y[len(Y)//2:]

# %%
model = linear_model.LinearRegression(fit_intercept=False)
model.fit(train2, trainY)
theta = model.coef_
print(theta)

# MSE and R^2
y_pred = model.predict(test2)

# Sum of squared errors (SSE)
sse = sum([x ** 2 for x in (testY - y_pred)])

test_mse2 = sse / len(testY)
test_mse2

# %%
model = linear_model.LinearRegression(fit_intercept=False)
model.fit(train3, trainY)
theta = model.coef_
print(theta)

# MSE and R^2
y_pred = model.predict(test3)

# Sum of squared errors (SSE)
sse = sum([x ** 2 for x in (testY - y_pred)])

test_mse3 = sse / len(testY)
test_mse3

# %%
answers['Q4'] = [test_mse2, test_mse3]

# %%
assertFloatList(answers['Q4'], 2)

# %%
### Question 5

# %%
f = open("beer_50000.json")
dataset = []
for l in f:
    dataset.append(eval(l))

# %%
dataset

# %%
X = [[1, len(d['review/text'])] for d in dataset]
y = [d['review/overall'] >= 4 for d in dataset]

# %%
X[:10]

# %%
model = linear_model.LogisticRegression(class_weight='balanced')
model.fit(X, y)
theta = model.coef_
theta

# %%
train_predictions = model.predict(X)

# %%
TP = sum([a == b for a,b in zip(train_predictions, y) if b == True])
TN = sum([a == b for a,b in zip(train_predictions, y) if b == False])
FP = sum([a != b for a,b in zip(train_predictions, y) if b == False])
FN = sum([a != b for a,b in zip(train_predictions, y) if b == True])

# %%
BER = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))

# %%
answers['Q5'] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q5'], 5)

# %%
### Question 6

# %%
scores = model.decision_function(X)
scores[:10]

# %%
scorelabels = list(zip(scores, y))
scorelabels.sort(reverse=True)
scorelabels[:10]

# %%
sortedlabels = [x[1] for x in scorelabels]
sortedlabels[:10]

# %%
precs = []

# %%
for k in [1,100,1000,10000]:
    precs.append(sum(sortedlabels[:k]) / k)

# %%
answers['Q6'] = precs

# %%
assertFloatList(answers['Q6'], 4)

# %%
### Question 7

# %%
dataset[0]

# %%
styles = set([d['beer/style'] for d in dataset])
len(styles)

# %%
years = set([d['review/timeStruct']['year'] for d in dataset])
len(years)

# %%
def feature(datum):
    length = len(datum['review/text'])
    year = datum['review/timeStruct']['year']
    interval = max(years) - min(years)
    year_encode = [0 for x in range(interval)]
    if year - 1999 > 0:
        year_encode[year-1999-1]=1
    
    month = datum['review/timeStruct']['mon']
    month_encode = [0 for x in range(12)]
    if month - 1 > 0:
        month_encode[month-1-1]=1
    
    return [1, length] + year_encode + month_encode

# %%
X = [feature(d) for d in dataset]
y = [d['review/overall'] >= 4 for d in dataset]

# %%
model = linear_model.LogisticRegression(class_weight='balanced')
model.fit(X, y)
theta = model.coef_
theta

# %%
train_predictions = model.predict(X)

# %%
TP = sum([a == b for a,b in zip(train_predictions, y) if b == True])
TN = sum([a == b for a,b in zip(train_predictions, y) if b == False])
FP = sum([a != b for a,b in zip(train_predictions, y) if b == False])
FN = sum([a != b for a,b in zip(train_predictions, y) if b == True])

# %%
BER2 = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
BER2

# %%
answers['Q7'] = ["I basically replicated the feature engineering part from \
regression to include more features from the dataset and then use \
one-hot encoding to incorporate the data into feature variable.", BER2]

# %%
f = open("answers_hw1.txt", 'w')
f.write(str(answers) + '\n')
f.close()


