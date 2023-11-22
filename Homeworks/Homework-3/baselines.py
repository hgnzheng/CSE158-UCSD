import gzip
from collections import defaultdict

def readJSON(path):
  for l in gzip.open(path, 'rt'):
    d = eval(l)
    u = d['userID']
    try:
      g = d['gameID']
    except Exception as e:
      g = None
    yield u,g,d

### Time-played baseline: compute averages for each user, or return the global average if we've never seen the user before

allHours = []
userHours = defaultdict(list)

for user,game,d in readJSON("train.json.gz"):
  h = d['hours_transformed']
  allHours.append(h)
  userHours[user].append(h)

globalAverage = sum(allHours) / len(allHours)
userAverage = {}
for u in userHours:
  userAverage[u] = sum(userHours[u]) / len(userHours[u])

predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,g = l.strip().split(',')
  if u in userAverage:
    predictions.write(u + ',' + g + ',' + str(userAverage[u]) + '\n')
  else:
    predictions.write(u + ',' + g + ',' + str(globalAverage) + '\n')

predictions.close()

### Would-play baseline: just rank which games are popular and which are not, and return '1' if a game is among the top-ranked

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
  if count > totalPlayed/2: break

predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,g = l.strip().split(',')
  if g in return1:
    predictions.write(u + ',' + g + ",1\n")
  else:
    predictions.write(u + ',' + g + ",0\n")

predictions.close()
