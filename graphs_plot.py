### @author  Anuja Pradeep Nagare 
###          anuja.nagare@uga.edu
### @version 25/04/2017

import time
start_time = time.time()

import pandas as pd

# read data
dataframe = pd.read_csv('scores.csv')

print(dataframe)
Scores_mean = dataframe['Scores_mean']
Scores_std = dataframe['Scores_std']

import numpy as np
import matplotlib.pyplot as plt

# red dashes, blue squares and green triangles
plt.plot(Scores_mean, 'r*')
plt.ylabel('Mean')
plt.show()

plt.plot(Scores_std, 'gs')
plt.ylabel('STD')

plt.show()
