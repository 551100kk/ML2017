import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, Cropping2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from preprocess import preprocess
from output import modify
import matplotlib.pyplot as plt


def first_model(path_train, path_test, istrain=False):
    train, test, id_test = preprocess(path_train, path_test)
    print('price changed done')

    y_train = np.log(train["price_doc"]).as_matrix()
    x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
    x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)

    num_train = len(x_train)
    x_all = pd.concat([x_train, x_test])



    for c in x_all.columns:
        if x_all[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_all[c].values))
            x_all[c] = lbl.transform(list(x_all[c].values))

    x_train = x_all[:num_train].as_matrix()
    x_test = x_all[num_train:].as_matrix()
    y_train = y_train.reshape((len(y_train), 1))
    



    for i in range(len(x_train)):
        for j in range(len(x_train[0])):
            if str(x_train[i][j]) == 'nan':
                x_train[i][j] = 0

    for i in range(len(x_test)):
        for j in range(len(x_test[0])):
            if str(x_test[i][j]) == 'nan':
                x_test[i][j] = 0

    print (x_train[0])
    print (y_train[0])


    model2 = Sequential()
    model2.add(Dense(input_dim=x_train.shape[1],output_dim=80))
    model2.add(Activation('relu'))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.1))
    model2.add(Dense(40))
    model2.add(Activation('relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(20))
    model2.add(Activation('relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(1))
    model2.summary()
    model2.compile(loss='mean_squared_error',optimizer="adam")

    history = model2.fit(x_train, y_train, epochs=251, batch_size=128, validation_split=0.2)

    # summarize history for loss
    '''plt.plot(history.history['loss'][5:])
    plt.plot(history.history['val_loss'][5:])
    plt.title('model loss')
    plt.ylabel('rmsle')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('process.png')'''
    
    y_predict = model2.predict(x_test).reshape( (len(x_test)) )
    return pd.DataFrame({'id': id_test, 'price_doc': np.exp(y_predict)})


def main():

    path_train  = '../input/train.csv'
    path_test   = '../input/test.csv'
    path_macro  = '../input/macro.csv'
    path_xgb    = 'xgb.csv'
    path_ans    = 'ans.csv'

    first_output = first_model(path_train, path_test)
    print ('First OK')

    result = first_output
    result.to_csv(path_xgb, index=False)
    print ('Output merged')



if __name__=='__main__':
    main()