###############################################################################
#                             importing packages                              #
###############################################################################
import pandas as pd
import numpy as np
import matplotlib

# import scikitlearn

###############################################################################
#                               importing data                                #
###############################################################################
df = pd.read_csv("~/Desktop/machine_learning/vgsales.csv")
df.shape
df.describe()
