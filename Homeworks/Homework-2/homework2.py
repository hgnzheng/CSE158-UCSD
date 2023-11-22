# %%
import random
from sklearn import linear_model
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
answers = {}

# %%
def parseData(fname):
    for l in open(fname):
        yield eval(l)

# %%
data = list(parseData("beer_50000.json"))

# %%
data[0]

# %%
random.seed(0)
random.shuffle(data)

# %%
# Already 50/25/25 split
dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]

# %%
yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]

# %%
categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1

# %%
categories = [c for c in categoryCounts if categoryCounts[c] > 1000]

# %%
catID = dict(zip(list(categories),range(len(categories))))

# %%


# %%
def feat(d):
    # In my solution, I wrote a reusable function that takes parameters to generate features for each question
    # Feel free to keep or discard
    cat = d['beer/style']
    ID = catID[cat] if cat in categories else 0
    # one-hot encoding
    encode = [0 for x in range(len(catID) -1)]
    if ID > 0:
        encode[ID - 1] = 1
    return [1] + encode

# %%
def pipeline(reg):
    X = [feat(d) for d in dataTrain]
    y = [d['beer/ABV'] > 7 for d in dataTrain]
    model = linear_model.LogisticRegression(C=reg, class_weight='balanced')
    model.fit(X, y)
    
    X_test = [feat(d) for d in dataTest]
    X_valid = [feat(d) for d in dataValid]
    y_test = [d['beer/ABV'] > 7 for d in dataTest]
    y_valid = [d['beer/ABV'] > 7 for d in dataValid]
    
    test_pred = model.predict(X_test)
    valid_pred = model.predict(X_valid)
    # Report the accuracy
    print(model.score(X_test, y_test))
    
    TP = sum([a == b for a,b in zip(test_pred, y_test) if b == True])
    TN = sum([a == b for a,b in zip(test_pred, y_test) if b == False])
    FP = sum([a != b for a,b in zip(test_pred, y_test) if b == False])
    FN = sum([a != b for a,b in zip(test_pred, y_test) if b == True])    
    
    BER_test = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_test)
    
    TP = sum([a == b for a,b in zip(valid_pred, y_valid) if b == True])
    TN = sum([a == b for a,b in zip(valid_pred, y_valid) if b == False])
    FP = sum([a != b for a,b in zip(valid_pred, y_valid) if b == False])
    FN = sum([a != b for a,b in zip(valid_pred, y_valid) if b == True])    
    
    BER_valid = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_valid)
    return model, BER_valid, BER_test

# %%
### Question 1

# %%
mod, validBER, testBER = pipeline(10)

# %%
answers['Q1'] = [validBER, testBER]
answers['Q1']

# %%
assertFloatList(answers['Q1'], 2)

# %%
### Question 2

# %%
dataTrain[0]

# %%
max_length = max([len(d['review/text']) for d in dataTrain])
max_length

# %%
def feat(d):
    cat = d['beer/style']
    ID = catID[cat] if cat in categories else 0
    # one-hot encoding
    encode = [0 for x in range(len(catID) -1)]
    if ID > 0:
        encode[ID - 1] = 1
    review_scores = [d['review/appearance'], d['review/palate'], d['review/taste'],
                    d['review/overall'], d['review/aroma']]
    
    return [1] + encode + review_scores + [len(d['review/text']) / max_length]

# %%
def pipeline(reg):
    X = [feat(d) for d in dataTrain]
    y = [d['beer/ABV'] > 7 for d in dataTrain]
    model = linear_model.LogisticRegression(C=reg, class_weight='balanced')
    model.fit(X, y)
    
    X_test = [feat(d) for d in dataTest]
    X_valid = [feat(d) for d in dataValid]
    y_test = [d['beer/ABV'] > 7 for d in dataTest]
    y_valid = [d['beer/ABV'] > 7 for d in dataValid]
    
    test_pred = model.predict(X_test)
    valid_pred = model.predict(X_valid)
    # Report the accuracy
    print(model.score(X_test, y_test))
    
    TP = sum([a == b for a,b in zip(test_pred, y_test) if b == True])
    TN = sum([a == b for a,b in zip(test_pred, y_test) if b == False])
    FP = sum([a != b for a,b in zip(test_pred, y_test) if b == False])
    FN = sum([a != b for a,b in zip(test_pred, y_test) if b == True])    
    
    BER_test = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_test)
    
    TP = sum([a == b for a,b in zip(valid_pred, y_valid) if b == True])
    TN = sum([a == b for a,b in zip(valid_pred, y_valid) if b == False])
    FP = sum([a != b for a,b in zip(valid_pred, y_valid) if b == False])
    FN = sum([a != b for a,b in zip(valid_pred, y_valid) if b == True])    
    
    BER_valid = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_valid)
    return model, BER_valid, BER_test

# %%
mod, validBER, testBER = pipeline(10)

# %%
answers['Q2'] = [validBER, testBER]
answers['Q2']

# %%
assertFloatList(answers['Q2'], 2)

# %%
### Question 3

# %%
naive_classifier = 0.5
valid_BER = 0.5
test_BER = 0.5
best_C = 0
model = None
for c in [0.001, 0.01, 0.1, 1, 10]:
    mod, validBER, testBER = pipeline(c)
    print(f"Validation BER for C={c} is {validBER}")
    mean_BER = (1/2) * (validBER + testBER)
    if mean_BER < naive_classifier:
        naive_classifier = mean_BER
        valid_BER = validBER
        test_BER = testBER
        best_C = c
        model = mod

# %%
bestC = best_C
bestC

# %%
mod, validBER, testBER = model, valid_BER, test_BER

# %%
answers['Q3'] = [bestC, validBER, testBER]
answers['Q3']

# %%
assertFloatList(answers['Q3'], 3)

# %%
### Question 4

# %%
def feat(d):
    cat = d['beer/style']
    ID = catID[cat] if cat in categories else 0
    # one-hot encoding
    encode = [0 for x in range(len(catID) -1)]
    if ID > 0:
        encode[ID - 1] = 1
    review_scores = [d['review/appearance'], d['review/palate'], d['review/taste'],
                    d['review/overall'], d['review/aroma']]
    
    return [1] + review_scores + [len(d['review/text']) / max_length]

def pipeline(reg):
    X = [feat(d) for d in dataTrain]
    y = [d['beer/ABV'] > 7 for d in dataTrain]
    model = linear_model.LogisticRegression(C=reg, class_weight='balanced')
    model.fit(X, y)
    
    X_test = [feat(d) for d in dataTest]
    X_valid = [feat(d) for d in dataValid]
    y_test = [d['beer/ABV'] > 7 for d in dataTest]
    y_valid = [d['beer/ABV'] > 7 for d in dataValid]
    
    test_pred = model.predict(X_test)
    valid_pred = model.predict(X_valid)
    # Report the accuracy
    print(model.score(X_test, y_test))
    
    TP = sum([a == b for a,b in zip(test_pred, y_test) if b == True])
    TN = sum([a == b for a,b in zip(test_pred, y_test) if b == False])
    FP = sum([a != b for a,b in zip(test_pred, y_test) if b == False])
    FN = sum([a != b for a,b in zip(test_pred, y_test) if b == True])    
    
    BER_test = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_test)
    
    TP = sum([a == b for a,b in zip(valid_pred, y_valid) if b == True])
    TN = sum([a == b for a,b in zip(valid_pred, y_valid) if b == False])
    FP = sum([a != b for a,b in zip(valid_pred, y_valid) if b == False])
    FN = sum([a != b for a,b in zip(valid_pred, y_valid) if b == True])    
    
    BER_valid = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_valid)
    return model, BER_valid, BER_test

# %%
mod, validBER, testBER_noCat = pipeline(1)

# %%
def feat(d):
    cat = d['beer/style']
    ID = catID[cat] if cat in categories else 0
    # one-hot encoding
    encode = [0 for x in range(len(catID) -1)]
    if ID > 0:
        encode[ID - 1] = 1
    review_scores = [d['review/appearance'], d['review/palate'], d['review/taste'],
                    d['review/overall'], d['review/aroma']]
    
    return [1] + encode + [len(d['review/text']) / max_length]

def pipeline(reg):
    X = [feat(d) for d in dataTrain]
    y = [d['beer/ABV'] > 7 for d in dataTrain]
    model = linear_model.LogisticRegression(C=reg, class_weight='balanced')
    model.fit(X, y)
    
    X_test = [feat(d) for d in dataTest]
    X_valid = [feat(d) for d in dataValid]
    y_test = [d['beer/ABV'] > 7 for d in dataTest]
    y_valid = [d['beer/ABV'] > 7 for d in dataValid]
    
    test_pred = model.predict(X_test)
    valid_pred = model.predict(X_valid)
    # Report the accuracy
    print(model.score(X_test, y_test))
    
    TP = sum([a == b for a,b in zip(test_pred, y_test) if b == True])
    TN = sum([a == b for a,b in zip(test_pred, y_test) if b == False])
    FP = sum([a != b for a,b in zip(test_pred, y_test) if b == False])
    FN = sum([a != b for a,b in zip(test_pred, y_test) if b == True])    
    
    BER_test = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_test)
    
    TP = sum([a == b for a,b in zip(valid_pred, y_valid) if b == True])
    TN = sum([a == b for a,b in zip(valid_pred, y_valid) if b == False])
    FP = sum([a != b for a,b in zip(valid_pred, y_valid) if b == False])
    FN = sum([a != b for a,b in zip(valid_pred, y_valid) if b == True])    
    
    BER_valid = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_valid)
    return model, BER_valid, BER_test

# %%
mod, validBER, testBER_noReview = pipeline(1)

# %%
def feat(d):
    cat = d['beer/style']
    ID = catID[cat] if cat in categories else 0
    # one-hot encoding
    encode = [0 for x in range(len(catID) -1)]
    if ID > 0:
        encode[ID - 1] = 1
    review_scores = [d['review/appearance'], d['review/palate'], d['review/taste'],
                    d['review/overall'], d['review/aroma']]
    
    return [1] + encode + review_scores

def pipeline(reg):
    X = [feat(d) for d in dataTrain]
    y = [d['beer/ABV'] > 7 for d in dataTrain]
    model = linear_model.LogisticRegression(C=reg, class_weight='balanced')
    model.fit(X, y)
    
    X_test = [feat(d) for d in dataTest]
    X_valid = [feat(d) for d in dataValid]
    y_test = [d['beer/ABV'] > 7 for d in dataTest]
    y_valid = [d['beer/ABV'] > 7 for d in dataValid]
    
    test_pred = model.predict(X_test)
    valid_pred = model.predict(X_valid)
    # Report the accuracy
    print(model.score(X_test, y_test))
    
    TP = sum([a == b for a,b in zip(test_pred, y_test) if b == True])
    TN = sum([a == b for a,b in zip(test_pred, y_test) if b == False])
    FP = sum([a != b for a,b in zip(test_pred, y_test) if b == False])
    FN = sum([a != b for a,b in zip(test_pred, y_test) if b == True])    
    
    BER_test = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_test)
    
    TP = sum([a == b for a,b in zip(valid_pred, y_valid) if b == True])
    TN = sum([a == b for a,b in zip(valid_pred, y_valid) if b == False])
    FP = sum([a != b for a,b in zip(valid_pred, y_valid) if b == False])
    FN = sum([a != b for a,b in zip(valid_pred, y_valid) if b == True])    
    
    BER_valid = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    # print(BER_valid)
    return model, BER_valid, BER_test

# %%
mod, validBER, testBER_noLength = pipeline(1)

# %%


# %%
answers['Q4'] = [testBER_noCat, testBER_noReview, testBER_noLength]
answers['Q4']

# %%
assertFloatList(answers['Q4'], 3)

# %%
### Question 5

# %%
path = "amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')

# %%
header

# %%
dataset = []

pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)

# %%
dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]

# %%
dataTrain[0]

# %%
# Feel free to keep or discard

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}
ratingDict = {} # To retrieve a rating for a specific user/item pair
reviewsPerUser = defaultdict(list)

for d in dataTrain:
    user,item = d['customer_id'], d['product_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    itemNames[item] = d['product_title']
    
for d in dataset:
    user,item = d['customer_id'], d['product_id']
    itemNames[item] = d['product_title']
    ratingDict[(user, item)] = d['star_rating']
    reviewsPerUser[user].append(d)

# %%
userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u, i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u, i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)

ratingMean = sum([d['star_rating'] for d in dataTrain]) / len(dataTrain)

# %%
dataTrain[0]

# %%
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom

# %%
def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim, i2))
    similarities.sort(reverse=True)
    return similarities[:N]

# %%
query = 'B00KCHRKD6'

# %%
ms = mostSimilar(query, 10)

# %%
answers['Q5'] = ms
answers['Q5']

# %%
assertFloatList([m[0] for m in ms], 10)

# %%
### Question 6

# %%
def MSE(y, ypred):
    differences = [(x-y)**2 for x,y in zip(ypred, y)]
    return sum(differences) / len(differences)

# %%
def predictRating(user,item):
    ratings = []
    similarities = []
    if item not in itemAverages:
        return ratingMean
    for d in reviewsPerUser[user]:
        i2 = d['product_id']
        if i2 == item: continue
        if i2 in itemAverages:
            ratings.append(d['star_rating'] - itemAverages[i2])
        else:
            ratings.append(d['star_rating'] - ratingMean)
        similarities.append(Jaccard(usersPerItem[item], usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings, similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        return itemAverages[item]

# %%
alwaysPredictMean = [ratingMean for d in dataTest]

# %%
simPredictions = [predictRating(d['customer_id'], d['product_id']) for d in dataTest]

# %%
labels = [d['star_rating'] for d in dataTest]

# %%
answers['Q6'] = MSE(simPredictions, labels)
answers['Q6']

# %%
assertFloat(answers['Q6'])

# %%
### Question 7

# %%
dataTrain[0]

# %%
from datetime import datetime
import math

# %%
def decay_function(time_1, time_2):
    return math.exp(-abs(time_1 - time_2))

# %%
def predictRatingV2(user,item, item_time):
    ratings = []
    similarities = []
    if item not in itemAverages:
        return ratingMean
    for d in reviewsPerUser[user]:
        i2 = d['product_id']
        i2_time = datetime.strptime(d['review_date'], "%Y-%m-%d").timestamp()
        item_time_stamp = datetime.strptime(item_time, "%Y-%m-%d").timestamp()
        if i2 == item: continue
        if i2 in itemAverages:
            ratings.append(d['star_rating'] - itemAverages[i2])
        else:
            ratings.append(d['star_rating'] - ratingMean)
        similarities.append(Jaccard(usersPerItem[item], usersPerItem[i2]) * decay_function(item_time_stamp, i2_time))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings, similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        return itemAverages[item]

# %%
simPredictions = [predictRatingV2(d['customer_id'], d['product_id'], d['review_date']) for d in dataTest]

# %%
labels = [d['star_rating'] for d in dataTest]

# %%
def MSE(y, ypred):
    differences = [(x-y)**2 for x,y in zip(ypred, y)]
    return sum(differences) / len(differences)

# %%
itsMSE = MSE(simPredictions, labels)
itsMSE

# %%
answers['Q7'] = ["The function basically takes in two timestamps calculated by the library. Then, I calculate the absolute difference between two times and then apply the result to the standard decay function model to obtain the weigh.", itsMSE]
answers['Q7']

# %%
assertFloat(answers['Q7'][1])

# %%
f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


