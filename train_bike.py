"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data_bike import process_data
from data.data_bike import process_data_wd
from model import model_t
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")
import time


import numpy as np
import keras
from keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, Convolution2D,LeakyReLU,ZeroPadding2D,MaxPooling2D,GlobalMaxPooling2D,Conv1D, Convolution1D, MaxPooling1D, Lambda, Add
from keras.models import Sequential, load_model

import keras.backend as K
from keras import layers

def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mae','msle'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('model/bike/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/bike/' + name + ' loss.csv', encoding='utf-8', index=False)

    font1 = {'family' : 'Times New Roman',
	    'weight' : 'normal',
	    'size'   : 13,
	     }

    plt.grid(ls='--')
    plt.xlabel('Number of Epochs',font1)
    plt.ylabel('Loss',font1)
    plt.tick_params(labelsize=10)
    plt.plot(hist.epoch, hist.history['loss'], 'red', lw=2,  label = 'Training')
    plt.plot(hist.epoch, hist.history['val_loss'], 'blue', lw=2, label = 'Validation')

    plt.legend()
    plt.savefig('model/bike/' + name +'_.eps', format='eps', dpi=1000)
    #plt.show()
    
    

def train_model_cross(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mae','msle'])
    hist = model.fit([X_train, X_train],y_train,epochs=config["epochs"], batch_size=config["batch"],validation_split=0.05)
    model.save('model/bike/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/bike/' + name + ' loss.csv', encoding='utf-8', index=False)
    plt.grid(ls='--')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.plot(hist.epoch, hist.history['loss'], 'red', lw=2,  label = 'Training')
    plt.plot(hist.epoch, hist.history['val_loss'], 'blue', lw=2, label = 'Validation')
    plt.legend()
    plt.savefig('model/bike/' + name +'_.eps', format='eps', dpi=1000)

    

def train_model_wd(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mae','msle'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    print("fit_np.atleast_3d(np.array(X_train)):",np.atleast_3d(np.array(X_train)).shape)
    hist = model.fit([np.atleast_3d(np.array(X_train)), np.atleast_3d(np.array(X_train))],np.atleast_3d(y_train),epochs=config["epochs"], batch_size=config["batch"],validation_split=0.05)
    #hist = model.fit(
    #    #[X_train,X_train], y_train,
    #    
    #    [np.atleast_3d(np.array(X_train)), np.atleast_3d(np.array(X_train))],np.atleast_3d(y_train),
    #    batch_size=config["batch"],
    #    epochs=config["epochs"],
    #    validation_split=0.05)
    ##result =model.fit([np.atleast_3d(np.array(train_x)), np.atleast_3d(np.array(train_x))],np.atleast_3d(train_y), nb_epoch=100, batch_size=80)
    model.save('model/bike/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/bike/' + name + ' loss.csv', encoding='utf-8', index=False)

    font1 = {'family' : 'Times New Roman',
	    'weight' : 'normal',
	    'size'   : 13,
	     }


    plt.grid(ls='--')
    plt.xlabel('Number of Epochs',font1)
    plt.ylabel('Loss',font1)
    plt.tick_params(labelsize=10)
    plt.plot(hist.epoch, hist.history['loss'], 'red', lw=2,  label = 'Training')
    plt.plot(hist.epoch, hist.history['val_loss'], 'blue', lw=2, label = 'Validation')

    plt.legend()
    plt.savefig('model/bike/' + name +'_.eps', format='eps', dpi=1000)
    #plt.show()
    
    

def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)






def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()

    lag = 12
    config = {"batch": 256, "epochs": 200}
    #file1 = 'data/train_bike.csv'
    #file2 = 'data/test_bike.csv'
    file1 = 'data/Fremont__SB__bicycle_count1.csv'
    file2 = 'data/Fremont__SB__bicycle_count2.csv'
    X_train, y_train, _, _, _ = process_data(file1, file2, lag)
    X_train_wd, y_train_wd, _, _, _ = process_data_wd(file1, file2, lag)
    print("inputdata_X_train_shape:",X_train.shape)

    if args.model == 'deep_reg':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model_t.deep_reg([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)

    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model_t.get_lstm([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)

    if args.model == 'lstm_attention':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        #print("X_train:",X_train)
        print("X_train_shape:",X_train.shape)
        m = model_t.get_lstm_attention([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    
    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model_t.get_gru([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model_t.get_saes([12, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config)

    if args.model == 'cnn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model_t.get_cnn()
        train_model(m, X_train, y_train, args.model, config)

    if args.model == 'cnn_lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model_t.cnn_lstm()
        train_model(m, X_train, y_train, args.model, config)

    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model_t.get_gru([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)

    if args.model == 'stdn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model_t.stdn(3, 3, 7, 3, 2)
        train_model(m, X_train, y_train, args.model, config)

    if args.model == 'lstm_cnn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        #print("X_train:",X_train)
        print("X_train_shape:",X_train.shape)
        m = model_t.lstm_cnn()
        train_model(m, X_train, y_train, args.model, config)

    if args.model == 'lstm_attention_cnn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        #print("X_train:",X_train)
        print("X_train_shape:",X_train.shape)
        m = model_t.lstm_attention_cnn()
        train_model(m, X_train, y_train, args.model, config)

    if args.model == 'wd':
        X_train_wd = np.reshape(X_train_wd, (X_train_wd.shape[0], X_train.shape[1], 1))
        #print("X_train:",X_train)
        #print("X_train_shape:",X_train_wd.shape)
        #print("np.atleast_3d(np.array(X_train)):",np.atleast_3d(np.array(X_train)).shape)
        #print("np.array(X_train):",np.array(X_train).shape)
        #print("y_train",y_train.shape)
        m = model_t.wd()
        train_model_wd(m, X_train_wd, y_train_wd, args.model, config)  

    if args.model == 'wd_crossLayer_attention':
        X_train_wd = np.reshape(X_train_wd, (X_train_wd.shape[0], X_train_wd.shape[1], 1))
        #print("X_train:",X_train)
        #print("X_train_shape:",X_train.shape)
        #print("np.atleast_3d(np.array(X_train)):",np.atleast_3d(np.array(X_train)).shape)
        #print("np.array(X_train):",np.array(X_train).shape)
        #print("y_train",y_train.shape)
        m = model_t.wd_crossLayer_attention()
        train_model_cross(m, X_train_wd, y_train_wd, args.model, config) 



    if args.model == 'w_attention_d':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        #print("X_train:",X_train)
        #print("X_train_shape:",X_train.shape)
        #print("np.atleast_3d(np.array(X_train)):",np.atleast_3d(np.array(X_train)).shape)
        #print("np.array(X_train):",np.array(X_train).shape)
        #print("y_train",y_train.shape)
        m = model_t.w_attention_d()
        train_model_wd(m, X_train_wd, y_train_wd, args.model, config) 

  

if __name__ == '__main__':
    starttime = time.time()
    main(sys.argv)
    endtime = time.time()
    dtime = endtime - starttime

    print("time: %.8s s" % dtime) 
