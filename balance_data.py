import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data.npy', allow_pickle=True)

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    prediction = data[1]

    if prediction == [1, 0, 0]:
        lefts.append([img, prediction])
    elif prediction == [0, 1, 0]:
        forwards.append([img, prediction])
    elif prediction == [0, 0, 1]:
        rights.append([img, prediction])
    else:
        print('no match')

forwards = forwards[:len(lefts)*2][:len(rights)*2]
lefts = lefts[:len(rights)]
rights = rights[:len(lefts)]

final_data = forwards + lefts + rights
shuffle(final_data)
np.save('training_data_v3.npy', final_data)