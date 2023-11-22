# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
import tensorflow as tf
from surprise.model_selection import train_test_split

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

# %%
answers = {}

# %%
# Some data structures that will be useful

# %%
allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)

# %%
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

# %%
##################################################
# Play prediction                                #
##################################################

# %%
hoursTrain[0]

# %%
# Any other preprocessing...
users = set() # maintain a set of unique users
games = set() # maintain a set of unique games
pos_pairs = []
for l in readGz("train.json.gz"):
    u,g = l['userID'], l['gameID']
    users.add(u)
    games.add(g)
    pos_pairs += [(u, g)]

# %%
train_data = pos_pairs[:165000]
valid_data = pos_pairs[165000:] # because pos_pairs has length 175000

# %%
len(users), len(games)

# %%
# Randomly sample games that weren't played
neg_pairs = set()
len_valid = len(hoursValid)
while True:
    user = random.sample(users, 1)[0]
    game = random.sample(games, 1)[0]
    if (user, game) not in pos_pairs:
        neg_pairs.add((user, game))
    if len(neg_pairs) == len_valid:
        break

# %%
len(neg_pairs), len(train_data)

# %%
### Question 1

# %%
# Evaluate baseline strategy
gameCount = defaultdict(int)
totalPlayed = 0

for user,game in train_data:
    gameCount[game] += 1
    totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPlayed/2: break

predictions = []
for user,game in valid_data:
    if game in return1:
        predictions += [1]
    else:
        predictions += [0]
    
for user,game in neg_pairs:
    if game in return1:
        predictions += [1]
    else:
        predictions += [0]

sum_pred = sum([predictions[i]==1 for i in range(9999)]) + \
    sum([predictions[i] == 0 for i in range(9999, len(predictions))])
acc = sum_pred / (2*len(valid_data))
acc

# %%
answers['Q1'] = acc
answers['Q1']

# %%
assertFloat(answers['Q1'])

# %%
### Question 2

# %%
# Improved strategy

# %%
# Evaluate baseline strategy
gameCount = defaultdict(int)
totalPlayed = 0

for user,game in train_data:
    gameCount[game] += 1
    totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPlayed/1.5: break # Choose 67% percentile instead

predictions = []
for user,game in valid_data:
    if game in return1:
        predictions += [1]
    else:
        predictions += [0]
    
for user,game in neg_pairs:
    if game in return1:
        predictions += [1]
    else:
        predictions += [0]

sum_pred_q2 = sum([predictions[i]==1 for i in range(9999)]) + \
    sum([predictions[i] == 0 for i in range(9999, len(predictions))])
acc_q2 = sum_pred_q2 / (2*len(valid_data))
acc_q2

# %%
answers['Q2'] = [1/1.5, acc_q2]
answers['Q2']

# %%
assertFloatList(answers['Q2'], 2)

# %%
### Question 3/4

# %%
userPerGame = defaultdict(set) # Maps a game to the users who played it
gamePerUser = defaultdict(set) # Maps a user to the game that they played
hoursDict = {} # To retrieve an hour for a specific user/game pair

for d in hoursTrain:
    user,game = d[0], d[1]
    userPerGame[game].add(user)
    gamePerUser[user].add(game)
    hoursDict[(user, game)] = d[2]['hours']

# %%
userAverages = {}
gameAverages = {}

for u in gamePerUser:
    rs = [hoursDict[(u,g)] for g in gamePerUser[u]]
    userAverages[u] = sum(rs) / len(rs)

for g in userPerGame:
    rs = [hoursDict[(u,g)] for u in userPerGame[g]]
    gameAverages[g] = sum(rs) / (len(rs))

# %%
reviewsPerUser = defaultdict(list)
reviewsPerGame = defaultdict(list)

for d in hoursTrain:
    user,game = d[0], d[1]
    reviewsPerUser[user].append(d)
    reviewsPerGame[game].append(d)

# %%
reviewsPerUser["u05450000"][0]

# %%
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom

def predictLabel(user, game, threshold):
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d[1]
        if i2 == game: continue
        similarities.append(Jaccard(userPerGame[game], userPerGame[i2]))
    return 1 if max(similarities) > threshold else 0

# %%
# Evaluate baseline strategy
predictions = []
for user,game in valid_data:
    predictions.append(predictLabel(user, game, 0.03))
# print(len(predictions))
for user,game in neg_pairs:
    predictions.append(predictLabel(user, game, 0.03))
# print(len(predictions))

sum_pred_q3 = sum([predictions[i]==1 for i in range(10000)]) + \
    sum([predictions[i] == 0 for i in range(10000, len(predictions))])
acc_q3 = sum_pred_q3 / (2*len(valid_data))
acc_q3

# %%
def threshold_and_popularity(threshold_popularity=1/1.5, threshold_jaccard=0.03):
    gameCount = defaultdict(int)
    totalPlayed = 0
    
    for user,game,_ in readJSON("train.json.gz"):
        gameCount[game] += 1
        totalPlayed += 1
        
    mostPopular = [(gameCount[x], x) for x in gameCount]
    mostPopular.sort()
    mostPopular.reverse()
    
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > threshold_popularity: break
    
    correct = 0
    for user, game in valid_data:
        similarities = []
        for d in reviewsPerUser[user]:
            i2 = d[1]
            if i2 == game: continue
            similarities.append(Jaccard(userPerGame[game], userPerGame[i2]))
            
        if max(similarities) > threshold_jaccard and game in return1:
            correct += (game in gamePerUser[user]) # recommend in this case
        else:
            correct += (game not in gamePerUser[user]) # not recommend in this case
        
    return correct/len(valid_data)

# %%
acc_q4 = threshold_and_popularity()
acc_q4

# %%
answers['Q3'] = acc_q3
answers['Q4'] = acc_q4

# %%
assertFloat(answers['Q3'])
assertFloat(answers['Q4'])

# %%
threshold_popularity = 1/1.5
threshold_jaccard = 0.003

gameCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in readJSON("train.json.gz"):
    gameCount[game] += 1
    totalPlayed += 1
    
mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPlayed*threshold_popularity: break

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > threshold_popularity: break

predictions = open("HWpredictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    userPlayed = gamePerUser[u]
    similarities = []
    # Logic...
    for d in reviewsPerUser[user]:
        i2 = d[1]
        if i2 == game: continue
        similarities.append(Jaccard(userPerGame[game], userPerGame[i2]))

    if max(similarities) > threshold_jaccard and game in return1:
        predictions.write(u + ',' + g + ',' + "1" + '\n')
    else:
        predictions.write(u + ',' + g + ',' + "0" + '\n')

predictions.close()

# %%
answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"

# %%
##################################################
# Hours played prediction                        #
##################################################

# %%
### Question 6

# %%
# I first tried the TF library, but it's not working...

# %%
# userIDs = {}
# itemIDs = {}
# interactions = []

# for d in allHours:
#     u = d[0]
#     i = d[1]
#     r = d[2]['hours_transformed']
#     if not u in userIDs: userIDs[u] = len(userIDs)
#     if not i in itemIDs: itemIDs[i] = len(itemIDs)
#     interactions.append((u,i,r))

# %%
# random.shuffle(interactions)
# len(interactions)

# %%
# nTrain = 165000
# nTest = len(interactions) - nTrain
# interactionsTrain = interactions[:nTrain]
# interactionsTest = interactions[nTrain:]

# %%
# itemsPerUser = defaultdict(list)
# usersPerItem = defaultdict(list)
# for u,i,r in interactionsTrain:
#     itemsPerUser[u].append(i)
#     usersPerItem[i].append(u)

# %%
# mu = sum([r for _,_,r in interactionsTrain] + [r for _,_,r in interactionsTest])*1.0 / len(allHours)
# mu

# %%
# optimizer = tf.keras.optimizers.Adam(0.1)

# %%
# class LatentFactorModel(tf.keras.Model):
#     def __init__(self, mu, lamb):
#         super(LatentFactorModel, self).__init__()
#         # Initialize to average
#         self.alpha = tf.Variable(mu)
#         # Initialize to small random values
#         self.betaU = tf.Variable(tf.zeros(len(userIDs)))
#         self.betaI = tf.Variable(tf.zeros(len(itemIDs)))
#         # self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
#         # self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
#         self.lamb = lamb

#     # Prediction for a single instance (useful for evaluation)
#     def predict(self, u, i):
#         p = self.alpha + self.betaU[u] + self.betaI[i] 
#         # + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
#         return p

#     # Regularizer
#     def reg(self):
#         return self.lamb * (tf.reduce_sum(self.betaU**2) +\
#                             tf.reduce_sum(self.betaI**2))
#                         # + tf.reduce_sum(self.gammaU**2) +\
#                         #     tf.reduce_sum(self.gammaI**2))
    
#     # Prediction for a sample of instances
#     def predictSample(self, sampleU, sampleI):
#         u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
#         i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
#         beta_u = tf.nn.embedding_lookup(self.betaU, u)
#         beta_i = tf.nn.embedding_lookup(self.betaI, i)
#         # gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
#         # gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
#         pred = self.alpha + beta_u + beta_i
#             #    tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
#         return pred
    
#     # Loss
#     def call(self, sampleU, sampleI, sampleR):
#         pred = self.predictSample(sampleU, sampleI)
#         r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
#         return tf.nn.l2_loss(pred - r) / len(sampleR)

# %%
# modelLFM = LatentFactorModel(mu, 1) # with lambda equal to 1

# %%
# modelLFM.trainable_variables[1]

# %%
# def trainingStep(model, interactions):
#     Nsamples = 50000
#     with tf.GradientTape() as tape:
#         sampleU, sampleI, sampleR = [], [], []
#         for _ in range(Nsamples):
#             u,i,r = random.choice(interactions)
#             sampleU.append(userIDs[u])
#             sampleI.append(itemIDs[i])
#             sampleR.append(r)

#         loss = model(sampleU,sampleI,sampleR)
#         loss += model.reg()
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients((grad, var) for
#                               (grad, var) in zip(gradients, model.trainable_variables)
#                               if grad is not None)
#     return loss.numpy()

# %%
# for i in range(100):
#     obj = trainingStep(modelLFM, interactionsTrain)
#     if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))

# %%
# predictions = []
# for u,g,r in interactionsTest:
#     predict = modelLFM.predict(userIDs[u], itemIDs[g]).numpy()
#     predictions.append(predict)
# len(predictions)

# %%
# def MSE(preds, labels):
#     diff = [(x-y)**2 for x,y in zip(preds, labels)]
#     return sum(diff) / len(diff)

# %%
# validlabels = [r for _,_,r in interactionsTest]
# validMSE = MSE(predictions,validlabels)
# validMSE

# %%
trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)

# %%
hoursPerUser, hoursPerItem,Rui = {}, {}, {}
for u,g,d in hoursTrain:
    if u not in hoursPerUser:
        hoursPerUser[u] = [(g, d['hours_transformed'])]
    else:
        hoursPerUser[u].append((g, d['hours_transformed']))
    if g not in hoursPerItem:
        hoursPerItem[g] = [(u, d['hours_transformed'])]
    else:
        hoursPerItem[g].append((u, d['hours_transformed']))
    Rui[(u,g)] = d['hours_transformed']

# %%
len(hoursPerItem)

# %%
betaU = {}
betaI = {}
for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0

# %%
alpha = globalAverage # Could initialize anywhere, this is a guess

# %%
def update_alpha():
    global alpha
    num = sum(Rui[(u,g)]- (betaU[u] + betaI[g]) for u,g,_ in hoursTrain)
    denom = len(hoursTrain)
    alpha = num / denom

# %%
def update_betaU(lamb):
    global alpha
    for user in hoursPerUser:
        num = sum(Rui[(user,g)]-(alpha+betaI[g]) for g,t in hoursPerUser[user])
        denom = lamb + len(hoursPerUser[user])
        betaU[user] = num / denom

# %%
def update_betaI(lamb):
    global alpha
    for item in hoursPerItem:
        num = sum(Rui[(u,item)]-(alpha+betaU[u]) for u,t in hoursPerItem[item])
        denom = lamb + len(hoursPerItem[item])
        betaI[item] = num / denom

# %%
def predict(user, item):
    global alpha
    if user not in hoursPerUser and item not in hoursPerItem:
        return alpha
    if user in hoursPerUser and item not in hoursPerItem:
        return alpha + betaU[user]
    if user not in hoursPerUser and item in hoursPerItem:
        return alpha + betaI[item]
    return alpha + betaU[user] + betaI[item]

# %%
def MSE():
    mse = sum((_['hours_transformed']-predict(u,i))**2 for u,i,_ in hoursValid) / len(hoursValid)
    return mse

# %%
def iterate(lamb, max_iteration=50):
    for iter in range(max_iteration):
        update_alpha()
        update_betaU(lamb)
        update_betaI(lamb)
        if iter % 10 == 9:
            validMSE = MSE()
            print(f"Current Iteration is {iter+1} | MSE: {validMSE}")
    return validMSE

# %%
validMSE = iterate(lamb=1)

# %%
answers['Q6'] = validMSE
answers['Q6']

# %%
assertFloat(answers['Q6'])

# %%
### Question 7

# %%
betaU

# %%
max_beta_u_id, max_beta_u = max(betaU, key=betaU.get), max(betaU.values())
min_beta_u_id, min_beta_u = min(betaU, key=betaU.get), min(betaU.values())
max_beta_I_id, max_beta_I = max(betaI, key=betaI.get), max(betaI.values())
min_beta_I_id, min_beta_I = min(betaI, key=betaI.get), min(betaI.values())


# %%
print("Maximum betaU = " + str(max_beta_u_id) + ' (' + str(max_beta_u) + ')')
print("Maximum betaI = " + str(max_beta_I_id) + ' (' + str(max_beta_I) + ')')
print("Minimum betaU = " + str(min_beta_u_id) + ' (' + str(min_beta_u) + ')')
print("Minimum betaI = " + str(min_beta_I_id) + ' (' + str(min_beta_I) + ')')

# %%
answers['Q7'] = [max_beta_u, min_beta_u, max_beta_I, min_beta_I]
answers['Q7']

# %%
assertFloatList(answers['Q7'], 4)

# %%
### Question 8

# %%
def iterate(lamb, max_iteration=50):
    for iter in range(max_iteration):
        update_alpha()
        update_betaU(lamb)
        update_betaI(lamb)
        if iter == 29:
            validMSE = MSE()
            print(f"Current Lambda is {lamb} | MSE: {validMSE}")
    return validMSE

# %%
# Better lambda...
bestValidMSE = None
bestLamb = 0
for lamb in np.arange(0, 10, 0.5):
    validMSE = iterate(lamb, max_iteration=30)
    if bestValidMSE == None or validMSE < bestValidMSE:
        bestValidMSE = validMSE
        bestLamb = lamb

# %%
answers['Q8'] = (bestLamb, bestValidMSE)
answers['Q8']

# %%
assertFloatList(answers['Q8'], 2)

# %%
predictions = open("HWpredictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    if u in betaU and g in betaI:
        predictions.write(u + ',' + g + ',' + str(alpha + betaU[u] + betaI[g]) + '\n')
    else:
        predictions.write(u + ',' + g + ',' + str(0) + '\n')

predictions.close()

# %%
f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



