import numpy as np
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout, Conv1D, MaxPooling1D, Flatten
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence
import os
import pickle

test_path = sys.argv[1]
output_path = sys.argv[2]

split_ratio = 0.0
nb_epoch = 20
batch_size = 128

def text_proc(text):
    return text

def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r', encoding = 'utf-8') as f:
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
            
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            text = text_proc(article).replace("'s", '').replace("n't", '')
            articles.append( text )
            # print (text.encode('utf8'))          
    return (tags,articles,tags_list)

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        #assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 

    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

thresh = 0.5
def f1_score(y_true,y_pred):
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))
def precision(y_true,y_pred):
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    return precision
def recall(y_true,y_pred):
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return recall

def get_bag(num_words, sequences):
    data = []
    for x in sequences:
        tmp = [0 for i in range(num_words)]
        for idx in x:
            tmp[idx] = 1
        data.append(tmp)
    return np.array(data)


def main():

    ### read training and testing data
    with open('tag_list.pkl', 'rb') as fp:
       tag_list = pickle.load(fp)
    (_, X_test,_) = read_data(test_path,False)
    with open('tokenizer.pkl', 'rb') as fp:
       tokenizer = pickle.load(fp)
    
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1

    print ('Convert to index sequences.')
    test_sequences = tokenizer.texts_to_sequences(X_test)

    print ('Get bag of words')
    test_sequences = get_bag(num_words, test_sequences)

    
    ### build model
    print ('Building model.')

    model = Sequential()
    model.add(Dense(512, input_dim=num_words, activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(len(tag_list),activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.001,decay=1e-5,clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score, precision, recall])

    # hist = model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size)
    # model.model.save_weights('best_model.hdf5')
    model.load_weights(os.path.join(os.path.dirname(__file__),'best_model.hdf5'))
    Y_pred = model.predict(test_sequences)
    
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
