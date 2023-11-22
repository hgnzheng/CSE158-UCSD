# %% [markdown]
# # CSE 158 Fall 2023, Assignment 1

# %% [markdown]
# ### Library Import

# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
from sklearn import linear_model
import numpy as np
import string
import random
import matplotlib.pyplot as plt
import tensorflow as tf

# %% [markdown]
# ### Utility Functions

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
allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)

# %%
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

# %%
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
for u,g,d in hoursTrain:
    r = d['hours_transformed']
    hoursPerUser[u].append((g, r))
    hoursPerItem[g].append((u, r))

# %%
itemsPerUser = defaultdict(list)
usersPerItem = defaultdict(list)
for u,g,d in hoursTrain:
    itemsPerUser[u].append(g)
    usersPerItem[g].append(u)

# %%
userIDs, itemIDs = {}, {}
interactions = []

for u,g,d in allHours:
    r = d['hours_transformed']
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not g in itemIDs: itemIDs[g] = len(itemIDs)
    interactions.append((u,g,r))

# %% [markdown]
# ## Part 1: Would-Play Prediction

# %%
# Generate a negative set
userSet = set()
gameSet = set()
playedSet = set()

for u,g,d in allHours:
    userSet.add(u)
    gameSet.add(g)
    playedSet.add((u, g))

lUserSet = list(userSet)
lGameSet = list(gameSet)

notPlayed = set()
for u,g,d in hoursValid:
    g = random.choice(lGameSet)
    while (u,g) in playedSet or (u,g) in notPlayed:
        g = random.choice(lGameSet)
    notPlayed.add((u,g))

playedValid = set()
for u,g,r in hoursValid:
    playedValid.add((u,g))

# %%
gameCount = defaultdict(int)
totalPlayed = 0

for u,g,_ in hoursTrain:
    gameCount[g] += 1
    totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPlayed/1.5: break

# %%
threshold = 1.4844999999999906

# %% [markdown]
# ### Bayesian Personalized Ranking Model

# %%
items = list(gameSet)

# %%
class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(gameSet)], stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userSet), K], stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(gameSet), K], stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb
    
    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p
    
    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI))
    
    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui
    
    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))

# %%
# Leverage powerful Adam optimizer with learning rate 0.1
optimizer = tf.keras.optimizers.Adam(0.1)

# %%
# Initialize the model
modelBPR = BPRbatch(5, 0.00001)

# %%
def trainingStepBPR(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [],[],[]
        for _ in range(Nsamples):
            u,i,_ = random.choice(interactions) # positive sample
            j = random.choice(items) # negative sample
            while j in itemsPerUser[u]:
                j = random.choice(items)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])
        
        loss = model(sampleU, sampleI, sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for 
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()

# %%
# Run 150 batches of gradient descent
for i in range(150):
    obj = trainingStepBPR(modelBPR, interactions)
    if (i % 10 == 9): print("Iteration " + str(i+1) + ", objective = " + str(obj))

# %% [markdown]
# ### Evaluation of the model

# %%
interactionsTrain = interactions[:165000]
interactionsTest = interactions[165000:]

# %%
interactionsTestPerUser = defaultdict(set)
itemSet = set()
for u,i,_ in interactionsTest:
    interactionsTestPerUser[u].add(i)
    itemSet.add(i)

# %%
def AUCu(u, N): # N samples per user
    win = 0
    if N > len(interactionsTestPerUser[u]):
        N = len(interactionsTestPerUser[u])
    positive = random.sample(interactionsTestPerUser[u],N)
    negative = random.sample(gameSet.difference(interactionsTestPerUser[u]),N)
    for i,j in zip(positive,negative):
        si = modelBPR.predict(userIDs[u], itemIDs[i]).numpy()
        sj = modelBPR.predict(userIDs[u], itemIDs[j]).numpy()
        if si > sj:
            win += 1
    return win/N

def AUC():
    av = []
    for u in interactionsTestPerUser:
        av.append(AUCu(u, 10))
    return sum(av) / len(av)

# %%
AUC()

# %% [markdown]
# ### Output Predictions

# %%
# Get games to predict for each user
pred_user_items = defaultdict(list)
for l in open('pairs_Played.csv'):
    if l.startswith("userID"):
        continue
    u,g = l.strip().split(',')
    if u in userIDs and g in itemIDs:
        score = modelBPR.predict(userIDs[u], itemIDs[g]).numpy()
    # If game is popular, recommend anyway
    elif g in return1:
        score = 100
    # If pair is unseen and game is not popular, would not recommend
    else:
        score = -100
    pred_user_items[u].append((g, score))

# %%
# Sort the game scores for each user
for key in pred_user_items:
    pred_user_items[key].sort(key=lambda x:x[1])
    pred_user_items[key].reverse()

# %%
# Check the data structure
pred_user_items

# %%
# Obtain positive and negative labels
positive_pairs = []
negative_pairs = []
for key in pred_user_items:
    for g,s in pred_user_items[key][:len(pred_user_items[key])//2]:
        positive_pairs.append((key,g))
    for g,s in pred_user_items[key][len(pred_user_items[key])//2:]:
        negative_pairs.append((key,g))

# %%
# Write to the output file.
predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    if (u,g) in positive_pairs:
        pred = 1
    else:
        pred = 0
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()

# %% [markdown]
# ## Part 2: Time-Played Prediction

# %%
Hours = [d['hours_transformed'] for u,g,d in allHours]
globalAverage = sum(Hours) * 1.0 / len(Hours)

# %%
hoursPerUser, hoursPerItem,Rui = {}, {}, {}
for u,g,d in allHours:
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
def iterate(lamb, max_iteration=1000):
    mse = 0
    for iter in range(max_iteration):
        update_alpha()
        update_betaU(lamb)
        update_betaI(lamb)
        curr_MSE = MSE()
        if iter % 10 == 9:
            print(f"Current Iteration is {iter+1} | MSE: {curr_MSE}")
        if mse == 0 or curr_MSE <= mse:
            mse = curr_MSE
        else:
            print(f"End Iteration is {iter+1} | MSE: {curr_MSE}")
            break
    return curr_MSE

# %%
# validMSE = iterate(lamb=5.02, max_iteration=500)

# %%
validMSE = iterate(lamb=5.0, max_iteration=500)

# %%
validMSE

# %% [markdown]
# 

# %% [markdown]
# ### Code block to tune lambda value

# %%
def iterate(lamb, max_iteration=50):
    mse = 0
    for iter in range(max_iteration):
        update_alpha()
        update_betaU(lamb)
        update_betaI(lamb)
        curr_mse = MSE()
        if mse == 0 or curr_mse <= mse:
            mse = curr_mse
        else:
            print(f"Current Lambda is {lamb} | MSE: {curr_mse}")
            break
        if iter == 29:
            print(f"Current Lambda is {lamb} | MSE: {curr_mse}")
    return mse

# %%
# Fine-tuning lambda
bestValidMSE = None
bestLamb = 0
for lamb in np.arange(4.9, 5.11, 0.01):
    validMSE = iterate(lamb, max_iteration=30)
    if bestValidMSE == None or validMSE < bestValidMSE:
        bestValidMSE = validMSE
        bestLamb = lamb

# %%
bestValidMSE, bestLamb # print the result

# %% [markdown]
# ### Output prediction

# %%
predictions = open("predictions_Hours.csv", 'w')
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

# %% [markdown]
# 


