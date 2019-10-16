"""
Defination of NN model
"""
import numpy as np
import keras
from keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, Convolution2D,LeakyReLU,ZeroPadding2D,MaxPooling2D,GlobalMaxPooling2D,Conv1D, Convolution1D, MaxPooling1D, Lambda, Add
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential, load_model
import attention
from attention import Attention
from keras_self_attention import SeqSelfAttention
import keras.backend as K
from keras import layers
from statsmodels.tsa.arima_model import ARIMA


def arima1(train):
    model = ARIMA(train, order=(1,1,1)) 
    result_arima = model.fit( disp=-1, method='css')

def arima(history,test):

    #X = ts.values
    #size = int(len(X) * 0.7)
    #train, test = X_train, y_train
    #history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(1, 2, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)



def deep_reg0():
    
    look_back = 12
    deep_data = Input(shape=(12,1))
    deep = Dense(10, activation='tanh', kernel_initializer='normal',
                input_dim=12)(deep_data)
    deep = Dropout(0.25)(deep)
    deep = Dense(5, activation='tanh',kernel_initializer='normal')(deep)
    deep = Dropout(0.25)(deep)
    deep = Dense(1)(deep)

    model = Model(inputs=deep_data, outputs=deep)


    print(model.summary())
    #model.compile(loss="mse", optimizer="adam") # adam, rmsprop
    return model




def deep_reg(units):
    model = Sequential()
    look_back = 12
    model.add(Dense(units[1], input_shape=(units[0], 1), activation='tanh', kernel_initializer='normal'))
    #model.add(Convolution1D(input_shape = (look_back,1), 
    #                    nb_filter=64,
    #                    filter_length=2,
    #                    border_mode='valid',
    #                    activation='relu',
    #                    subsample_length=1)) 
    model.add(MaxPooling1D(pool_length=2)) 

    model.add(Dropout(0.25)) 
    model.add(Flatten()) 
    model.add(Dense(units[2])) 
    model.add(Dropout(0.25))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('linear'))

    return model




def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))
    print(model.summary())
    return model


def get_cnn():
    model = Sequential()
    look_back = 12
    model.add(Convolution1D(input_shape = (look_back,1), 
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)) 
    model.add(Conv1D(128, 1, padding='valid'))
    model.add(MaxPooling1D(pool_length=2)) 

    model.add(Dropout(0.25)) 
    model.add(Flatten()) 
    model.add(Dense(250)) 
    model.add(Dropout(0.25))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('linear'))

    return model


def get_lstm_attention(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    #model.add(LSTM(units[2]))
    
    model.add(Convolution1D(input_shape = (12,1),
                        nb_filter=units[2],# 32,128
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
    #model.add(Conv1D(64, 1, padding='valid'))
    #model.add(Conv1D(64,1))
    model.add(LSTM(units[2]))

    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model



def lstm_cnn():
    # Construct the whole LSTM + CNN
    model = Sequential()

    look_back = 12
    # LSTM
    model.add(LSTM(12, input_shape = (look_back, 1), return_sequences=True))
    #model.add(LSTM(64))
    # CNN
    model.add(Convolution1D(input_shape = (look_back,1),
                        nb_filter=128,# 32,128
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
    model.add(Conv1D(128, 1, padding='valid'))

    model.add(MaxPooling1D(pool_length=2))
    model.add(LSTM(64))
    model.add(Dropout(0.25))
    #model.add(Flatten()) 
    model.add(Activation('relu')) # ReLU : y = max(0,x)
    model.add(Dense(1))
    model.add(Activation('sigmoid')) # Linear : y = x

    # Print whole structure of the model
    print(model.summary())
    return model

def wd():
    look_back = 12
    wide = Input(shape=(look_back, 1))
    print("wide_shape",wide.shape)
    deep_data = Input(shape=(look_back, 1))
    print("deep_data_shape",deep_data.shape)
    deep = Dense(input_dim=1, output_dim=6, activation='relu')(deep_data)
    deep = Dense(128, activation='relu')(deep)
    print("deep_shape",deep.shape)



   
   # wide & deep 
    wide_deep = concatenate([wide, deep])
    print("wide_deep_shape",wide_deep.shape)
    wide_deep = Dense(1, activation='sigmoid')(wide_deep)
    print("wide_deep2_shape",wide_deep.shape)
    model = Model(inputs=[wide, deep_data], outputs=wide_deep)


    print(model.summary())
    #model.compile(loss="mse", optimizer="adam") # adam, rmsprop
    return model


def wdcnn():
    look_back = 12
    wide = Input(shape=(look_back, 1))
    print("wide_shape",wide.shape)
    deep_data = Input(shape=(look_back, 1))
    print("deep_data_shape",deep_data.shape)
    deep = Dense(input_dim=1, output_dim=6, activation='relu')(deep_data)
    deep = Conv1D(128,1)(deep)
    #deep = Dense(128, activation='relu')(deep)
    print("deep_shape",deep.shape)



   
   # wide & deep 
    wide_deep = concatenate([wide, deep])
    print("wide_deep_shape",wide_deep.shape)
    wide_deep = Dense(1, activation='sigmoid')(wide_deep)
    print("wide_deep2_shape",wide_deep.shape)
    model = Model(inputs=[wide, deep_data], outputs=wide_deep)


    print(model.summary())
    #model.compile(loss="mse", optimizer="adam") # adam, rmsprop
    return model

def wd_attention():
    look_back = 12
    wide = Input(shape=(look_back, 1))
    print("wide_shape",wide.shape)
    deep_data = Input(shape=(look_back, 1))
    print("deep_data_shape",deep_data.shape)
    deep = Dense(input_dim=1, output_dim=6, activation='relu')(deep_data)
    #deep = Conv1D(128,1)(deep)
    #deep = Dense(128, activation='relu')(deep)
    print("deep_shape",deep.shape)
    deep = LSTM(12, return_sequences=True)(deep)


    deep = SeqSelfAttention(attention_activation='sigmoid', name='Attention')(deep)


    deep = Conv1D(128,1)(deep)
   
   # wide & deep 
    wide_deep = concatenate([wide, deep])
    print("wide_deep_shape",wide_deep.shape)
    wide_deep = Dense(1, activation='sigmoid')(wide_deep)
    print("wide_deep2_shape",wide_deep.shape)
    model = Model(inputs=[wide, deep_data], outputs=wide_deep)


    print(model.summary())
    #model.compile(loss="mse", optimizer="adam") # adam, rmsprop
    return model


def wd_re_attention():
    look_back = 12
    wide = Input(shape=(look_back, 1))
    print("wide_shape",wide.shape)
    deep_data = Input(shape=(look_back, 1))
    print("deep_data_shape",deep_data.shape)
    deep = Dense(input_dim=1, output_dim=6, activation='relu')(deep_data)
    #deep = Conv1D(128,1)(deep)
    #deep = Dense(128, activation='relu')(deep)
    print("deep_shape",deep.shape)
    deep = LSTM(12, return_sequences=True)(deep)


    deep = SeqSelfAttention(kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation='sigmoid', name='Attention')(deep)
                       #attention_regularizer_weight=1e-4,attention_activation='sigmoid', name='Attention')(deep)


    deep = Conv1D(128,1)(deep)
   
   # wide & deep 
    wide_deep = concatenate([wide, deep])
    print("wide_deep_shape",wide_deep.shape)
    wide_deep = Dense(1, activation='sigmoid')(wide_deep)
    print("wide_deep2_shape",wide_deep.shape)
    model = Model(inputs=[wide, deep_data], outputs=wide_deep)


    print(model.summary())
    #model.compile(loss="mse", optimizer="adam") # adam, rmsprop
    return model



def wd_new_attention():
    look_back = 12
    #wide = Input(shape=(look_back, 1))
    wide_data = Input(shape=(look_back, 1))
    #print("wide_shape",wide.shape)
    #wide = Dense(input_dim=1, output_dim=6, activation='sigmoid')(wide_data)
    wide = Conv1D(128,1,activation='relu')(wide_data)
    print("wide2_shape",wide.shape)
    

    deep_data = Input(shape=(look_back, 1))
    print("deep_data_shape",deep_data.shape)
    deep = Dense(input_dim=1, output_dim=6, activation='relu')(deep_data)
    #deep = Conv1D(128,1)(deep)
    #deep = Dense(128, activation='relu')(deep)
    print("deep_shape",deep.shape)
    deep = LSTM(12, return_sequences=True)(deep)


    deep = SeqSelfAttention(kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation='sigmoid', name='Attention')(deep)
                       #attention_regularizer_weight=1e-4,attention_activation='sigmoid', name='Attention')(deep)


    deep = Conv1D(128,1)(deep)
   
   # wide & deep 
    wide_deep = concatenate([wide, deep])
    print("wide_deep_shape",wide_deep.shape)
    wide_deep = Dense(1, activation='sigmoid')(wide_deep)
    print("wide_deep2_shape",wide_deep.shape)
    model = Model(inputs=[wide_data, deep_data], outputs=wide_deep)


    print(model.summary())
    #model.compile(loss="mse", optimizer="adam") # adam, rmsprop
    return model

def w_attention_d():
    look_back = 12
    #wide = Input(shape=(look_back, 1))
    wide_data = Input(shape=(look_back, 1))
    wide = SeqSelfAttention(kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation='sigmoid', name='Attention')(wide_data)
    #print("wide_shape",wide.shape)
    #wide = Dense(input_dim=1, output_dim=6, activation='sigmoid')(wide_data)
    #wide = Conv1D(128,1,activation='relu')(wide_data)
    print("wide2_shape",wide.shape)
    

    deep_data = Input(shape=(look_back, 1))
    print("deep_data_shape",deep_data.shape)
    deep = Dense(input_dim=1, output_dim=6, activation='relu')(deep_data)
    #deep = Conv1D(128,1)(deep)
    #deep = Dense(128, activation='relu')(deep)
    print("deep_shape",deep.shape)
    deep = LSTM(12, return_sequences=True)(deep)


    #deep = SeqSelfAttention(kernel_regularizer=keras.regularizers.l2(1e-4),
                       #bias_regularizer=keras.regularizers.l1(1e-4),
                       #attention_activation='sigmoid', name='Attention')(deep)
                       #attention_regularizer_weight=1e-4,attention_activation='sigmoid', name='Attention')(deep)


    deep = Conv1D(128,1)(deep)
   
   # wide & deep 
    wide_deep = concatenate([wide, deep])
    print("wide_deep_shape",wide_deep.shape)
    wide_deep = Dense(1, activation='sigmoid')(wide_deep)
    print("wide_deep2_shape",wide_deep.shape)
    model = Model(inputs=[wide_data, deep_data], outputs=wide_deep)


    print(model.summary())
    #model.compile(loss="mse", optimizer="adam") # adam, rmsprop
    return model


def w_attention_d_parm():
    look_back = 12
    wide_data = Input(shape=(look_back, 1))
    wide = SeqSelfAttention(
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation='sigmoid', name='Attention')(wide_data)
    print("wide2_shape",wide.shape)
    

    deep_data = Input(shape=(look_back, 1))
    print("deep_data_shape",deep_data.shape)
    deep = Dense(input_dim=1, output_dim=6, activation='relu')(deep_data)
    print("deep_shape",deep.shape)
    deep = LSTM(8, return_sequences=True)(deep)
    deep = Conv1D(128,1)(deep)
   
   # wide & deep 
    wide_deep = concatenate([wide, deep])
    print("wide_deep_shape",wide_deep.shape)
    wide_deep = Dense(1, activation='sigmoid')(wide_deep)
    print("wide_deep2_shape",wide_deep.shape)
    model = Model(inputs=[wide_data, deep_data], outputs=wide_deep)
    print(model.summary())

    return model


class CrossLayer(layers.Layer):
    def __init__(self, **kwargs):
        self.output_dim = 12
        self.num_layer = 2
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[2]
        #self.input_dim = 12
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape = [1, self.input_dim], initializer = 'glorot_uniform', name = 'w_' + str(i), trainable = True))
            self.bias.append(self.add_weight(shape = [1, self.input_dim], initializer = 'zeros', name = 'b_' + str(i), trainable = True))
        self.built = True

    def call(self, input):
        for i in range(self.num_layer):
            if i == 0:
                
                cross = Lambda(lambda x: Add()([K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 12)), x), 1, keepdims = True), self.bias[i], x]))(input)
            else:
                cross = Lambda(lambda x: Add()([K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 12)), input), 1, keepdims = True), self.bias[i], input]))(cross)

        
        return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)


def wd_crossLayer_attention():
    look_back = 12
    #wide = Input(shape=(look_back, 1))
    wide_data = Input(shape=(look_back, 1))
    #print("wide_shape",wide.shape)
    #wide = Dense(input_dim=1, output_dim=6, activation='sigmoid')(wide_data)
    #wide = Conv1D(128,1,activation='relu')(wide_data)
    #cross = CrossLayer(output_dim = 12, num_layer = 1, name = "cross_layer")(wide_data)
    cross = CrossLayer( name = "cross_layer")(wide_data)
    print("crosss_shape:",cross.shape)
    #print("wide2_shape",wide.shape)
    

    deep_data = Input(shape=(look_back, 1))
    print("deep_data_shape",deep_data.shape)
    deep = Dense(input_dim=1, output_dim=6, activation='relu')(deep_data)
    #deep = Conv1D(128,1)(deep)
    #deep = Dense(128, activation='relu')(deep)
    print("deep_shape",deep.shape)
    #deep = LSTM(12)(deep)
    deep = LSTM(12, return_sequences=True)(deep)


    deep = SeqSelfAttention(kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_activation='sigmoid', name='Attention')(deep)
                       #attention_regularizer_weight=1e-4,attention_activation='sigmoid', name='Attention')(deep)


    deep = LSTM(12)(deep)
    
   
   # wide & deep 
    wide_deep = concatenate([cross, deep])
    print("wide_deep_shape",wide_deep.shape)
    wide_deep = Dense(12, activation='sigmoid')(wide_deep)
    print("wide_deep2_shape",wide_deep.shape)
    model = Model(inputs=[wide_data, deep_data], outputs=wide_deep)


    print(model.summary())
    #model.compile(loss="mse", optimizer="adam") # adam, rmsprop
    return model



def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models







def get_lstm_cnn():
    n_classes = 1
    inp=Input(shape=(12, 1))
    reshape=Reshape((1,12,1))(inp)
 #   pre=ZeroPadding2D(padding=(1, 1))(reshape)
    # 1
    conv1=Convolution2D(32, 3, 3, border_mode='same',init='glorot_uniform')(reshape)
    #model.add(Activation('relu'))
    l1=LeakyReLU(alpha=0.33)(conv1)
 
    conv2=ZeroPadding2D(padding=(1, 1))(l1)
    conv2=Convolution2D(32, 3, 3, border_mode='same',init='glorot_uniform')(conv2)
    #model.add(Activation('relu'))
    l2=LeakyReLU(alpha=0.33)(conv2)
 
    m2=MaxPooling2D((3, 3), strides=(3, 3))(l2)
    d2=Dropout(0.25)(m2)
    # 2
    conv3=ZeroPadding2D(padding=(1, 1))(d2)
    conv3=Convolution2D(64, 3, 3, border_mode='same',init='glorot_uniform')(conv3)
    #model.add(Activation('relu'))
    l3=LeakyReLU(alpha=0.33)(conv3)
 
    conv4=ZeroPadding2D(padding=(1, 1))(l3)
    conv4=Convolution2D(64, 3, 3, border_mode='same',init='glorot_uniform')(conv4)
    #model.add(Activation('relu'))
    l4=LeakyReLU(alpha=0.33)(conv4)
 
    m4=MaxPooling2D((3, 3), strides=(3, 3))(l4)
    d4=Dropout(0.25)(m4)
    # 3
    conv5=ZeroPadding2D(padding=(1, 1))(d4)
    conv5=Convolution2D(128, 3, 3, border_mode='same',init='glorot_uniform')(conv5)
    #model.add(Activation('relu'))
    l5=LeakyReLU(alpha=0.33)(conv5)
 
    conv6=ZeroPadding2D(padding=(1, 1))(l5)
    conv6=Convolution2D(128, 3, 3, border_mode='same',init='glorot_uniform')(conv6)
    #model.add(Activation('relu'))
    l6=LeakyReLU(alpha=0.33)(conv6)
 
    m6=MaxPooling2D((3, 3), strides=(3, 3))(l6)
    d6=Dropout(0.25)(m6)
    # 4
    conv7=ZeroPadding2D(padding=(1, 1))(d6)
    conv7=Convolution2D(256, 3, 3, border_mode='same',init='glorot_uniform')(conv7)
    #model.add(Activation('relu'))
    l7=LeakyReLU(alpha=0.33)(conv7)
 
    conv8=ZeroPadding2D(padding=(1, 1))(l7)
    conv8=Convolution2D(256, 3, 3, border_mode='same',init='glorot_uniform')(conv8)
    #model.add(Activation('relu'))
    l8=LeakyReLU(alpha=0.33)(conv8)
    g=GlobalMaxPooling2D()(l8)
    print("g=",g)
    #g1=Flatten()(g)
    lstm1=LSTM(
        input_shape=(12,1),
        output_dim=1,
        activation='tanh',
        return_sequences=False)(inp)
    dl1=Dropout(0.3)(lstm1)
    
    den1=Dense(200,activation="relu")(dl1)
    #model.add(Activation('relu'))
    #l11=LeakyReLU(alpha=0.33)(d11)
    dl2=Dropout(0.3)(den1)
 
#     lstm2=LSTM(
#         256,activation='tanh',
#         return_sequences=False)(lstm1)
#     dl2=Dropout(0.5)(lstm2)
    print("dl2=",dl1)
#    global g2
    g2=concatenate([g,dl2],axis=1)
    d10=Dense(1024)(g2)
    #model.add(Activation('relu'))
    l10=LeakyReLU(alpha=0.33)(d10)
    l10=Dropout(0.5)(l10)
    l11=Dense(n_classes, activation='softmax')(l10)
 
 
 
    model=Model(input=inp,outputs=l11)
    model.summary()
    #complie model
    adam = keras.optimizers.Adam(lr = 0.0005, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    #adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    #sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)
 
    #reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
 
    
    return model




def stdn(self, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, cnn_flat_size = 128, lstm_out_size = 128,\
    nbhd_size = 3, nbhd_type = 2, map_x_num = 10, map_y_num = 20, flow_type = 4, output_shape = 2, optimizer = 'adagrad', loss = 'mse', metrics=[]):
        flatten_att_nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "att_nbhd_volume_input_time_{0}_{1}".format(att+1, ts+1)) for ts in range(att_lstm_seq_len) for att in range(att_lstm_num)]
        flatten_att_flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "att_flow_volume_input_time_{0}_{1}".format(att+1, ts+1)) for ts in range(att_lstm_seq_len) for att in range(att_lstm_num)]

        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])

        att_lstm_inputs = [Input(shape = (att_lstm_seq_len, feature_vec_len,), name = "att_lstm_input_{0}".format(att+1)) for att in range(att_lstm_num)]
        nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "nbhd_volume_input_time_{0}".format(ts+1)) for ts in range(lstm_seq_len)]
        flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "flow_volume_input_time_{0}".format(ts+1)) for ts in range(lstm_seq_len)]
        lstm_inputs = Input(shape = (lstm_seq_len, feature_vec_len,), name = "lstm_input")

        #short-term part
        #1st level gate
        #nbhd cnn
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time0_{0}".format(ts+1))(nbhd_inputs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time0_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        #flow cnn
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time0_{0}".format(ts+1))(flow_inputs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time0_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        #flow gate
        flow_gates = [Activation("sigmoid", name = "flow_gate0_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]


        #2nd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time1_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time1_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time1_{0}".format(ts+1))(flow_inputs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time1_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate1_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]

        #3rd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time2_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time2_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time2_{0}".format(ts+1))(flow_inputs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time2_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate2_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]


        #dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_vecs = [Dense(units = cnn_flat_size, name = "nbhd_dense_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(lstm_seq_len)]

        #feature concatenate
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (lstm_seq_len, cnn_flat_size))(nbhd_vec)
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])

        #lstm
        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

        #attention part
        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_inputs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time0_{0}_{1}".format(att+1,ts+1))(att_flow_inputs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate0_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate1_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate2_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Dense(units = cnn_flat_size, name = "att_nbhd_dense_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]


        att_nbhd_vec = [Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(att_lstm_num)]
        att_nbhd_vec = [Reshape(target_shape = (att_lstm_seq_len, cnn_flat_size))(att_nbhd_vec[att]) for att in range(att_lstm_num)]
        att_lstm_input = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att]]) for att in range(att_lstm_num)]

        att_lstms = [LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_input[att]) for att in range(att_lstm_num)]

        #compare
        att_low_level=[attention.Attention(method='cba')([att_lstms[att], lstm]) for att in range(att_lstm_num)]
        att_low_level=Concatenate(axis=-1)(att_low_level)
        att_low_level=Reshape(target_shape=(att_lstm_num, lstm_out_size))(att_low_level)


        att_high_level = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = Dense(units = output_shape)(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        inputs = flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + nbhd_inputs + flow_inputs + [lstm_inputs,]
        # print("Model input length: {0}".format(len(inputs)))
        # ipdb.set_trace()
        model = Model(inputs = inputs, outputs = pred_volume)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model
