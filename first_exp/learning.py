###############################################################################
#                             importing packages                              #
###############################################################################
import pandas as pd
import numpy as np
import matplotlib
from sklearn.tree import DecisionTreeClassifier as dtc

###############################################################################
#                               importing data                                #
###############################################################################
df = pd.read_csv("~/Desktop/machine_learning/first_exp/data/vgsales.csv")

################
#  first look  #
################
df.shape
df.describe()
df.values

df.hist()

###############################################################################
#                                 music task                                  #
###############################################################################
music_data = pd.read_csv("~/Desktop/machine_learning/first_exp/data/music.csv")

####################################################################
#  splitting the data set into an input(X), and an output(Y) one   #
####################################################################
X = music_data.drop(columns=["genre"])
Y = music_data["genre"]

model = dtc()
model.fit(X, Y)
predictions = model.predict([[21, 1], [22, 0]])
predictions
# Out[28]: array(['HipHop', 'Dance'], dtype=object)
# says that a 21 year old male would prefer HipHop, whereas the 22 year old
# female would prefer to listen to dance music
