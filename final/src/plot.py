import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dnn = pd.read_csv('dnn.csv')["price_doc"].as_matrix()[:1000]
xgb = pd.read_csv('best.csv')["price_doc"].as_matrix()[:1000]
train = pd.read_csv('../input/train.csv')["price_doc"].as_matrix()


plt.figure(figsize=(15, 5))
plt.plot(train)
#plt.plot(dnn)
#plt.plot(xgb)
plt.title('Output Camparison')
plt.ylabel('Price')
plt.xlabel('id')
plt.legend(['train'], loc='upper left')
plt.savefig('compare_dnn.png')