from keras.models import Model
from keras.layers import (
    Conv2D,
    Dropout,
    Input,
    Activation,
    concatenate,
    MaxPooling2D,
    Conv2DTranspose,
)


def encoder_block(inputs, n_filters, dropout_rate):
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(inputs)
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = Activation('relu')(x)
    f = x
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_rate)(x)

    return x, f


def encoder(inputs):
    x, f1 = encoder_block(inputs, n_filters=64, dropout_rate=0.3)
    x, f2 = encoder_block(x, n_filters=128, dropout_rate=0.3)
    x, f3 = encoder_block(x, n_filters=256, dropout_rate=0.3)
    x, f4 = encoder_block(x, n_filters=512, dropout_rate=0.3)
    return x, (f1, f2, f3, f4)


def bottleneck(inputs):
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(inputs)
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
    x = Activation('relu')(x)
    return x


def decoder_block(inputs, f, n_filters, dropout_rate):
    x = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=2, padding='same')(inputs)
    x = concatenate([x, f])
    x = Dropout(rate=dropout_rate)(x)
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = Activation('relu')(x)
    return x


def decoder(inputs, convs):
    f1, f2, f3, f4 = convs
    x = decoder_block(inputs, f4, n_filters=512, dropout_rate=0.3)
    x = decoder_block(x, f3, n_filters=256, dropout_rate=0.3)
    x = decoder_block(x, f2, n_filters=128, dropout_rate=0.3)
    x = decoder_block(x, f1, n_filters=64, dropout_rate=0.3)

    outputs = Conv2D(filters=3, kernel_size=(1, 1), activation='softmax')(x)

    return outputs


def Unet():
    inputs = Input(shape=(128, 128, 3,))

    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs)

    model = Model(inputs, outputs)

    return model


unet = Unet()
unet.summary()

"""
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 128, 128, 3  0           []                               
                                )]                                                                
                                                                                                  
 conv2d (Conv2D)                (None, 128, 128, 64  1792        ['input_1[0][0]']                
                                )                                                                 
                                                                                                  
 conv2d_1 (Conv2D)              (None, 128, 128, 64  36928       ['conv2d[0][0]']                 
                                )                                                                 
                                                                                                  
 activation (Activation)        (None, 128, 128, 64  0           ['conv2d_1[0][0]']               
                                )                                                                 
                                                                                                  
 max_pooling2d (MaxPooling2D)   (None, 64, 64, 64)   0           ['activation[0][0]']             
                                                                                                  
 dropout (Dropout)              (None, 64, 64, 64)   0           ['max_pooling2d[0][0]']          
                                                                                                  
 conv2d_2 (Conv2D)              (None, 64, 64, 128)  73856       ['dropout[0][0]']                
                                                                                                  
 conv2d_3 (Conv2D)              (None, 64, 64, 128)  147584      ['conv2d_2[0][0]']               
                                                                                                  
 activation_1 (Activation)      (None, 64, 64, 128)  0           ['conv2d_3[0][0]']               
                                                                                                  
 max_pooling2d_1 (MaxPooling2D)  (None, 32, 32, 128)  0          ['activation_1[0][0]']           
                                                                                                  
 dropout_1 (Dropout)            (None, 32, 32, 128)  0           ['max_pooling2d_1[0][0]']        
                                                                                                  
 conv2d_4 (Conv2D)              (None, 32, 32, 256)  295168      ['dropout_1[0][0]']              
                                                                                                  
 conv2d_5 (Conv2D)              (None, 32, 32, 256)  590080      ['conv2d_4[0][0]']               
                                                                                                  
 activation_2 (Activation)      (None, 32, 32, 256)  0           ['conv2d_5[0][0]']               
                                                                                                  
 max_pooling2d_2 (MaxPooling2D)  (None, 16, 16, 256)  0          ['activation_2[0][0]']           
                                                                                                  
 dropout_2 (Dropout)            (None, 16, 16, 256)  0           ['max_pooling2d_2[0][0]']        
                                                                                                  
 conv2d_6 (Conv2D)              (None, 16, 16, 512)  1180160     ['dropout_2[0][0]']              
                                                                                                  
 conv2d_7 (Conv2D)              (None, 16, 16, 512)  2359808     ['conv2d_6[0][0]']               
                                                                                                  
 activation_3 (Activation)      (None, 16, 16, 512)  0           ['conv2d_7[0][0]']               
                                                                                                  
 max_pooling2d_3 (MaxPooling2D)  (None, 8, 8, 512)   0           ['activation_3[0][0]']           
                                                                                                  
 dropout_3 (Dropout)            (None, 8, 8, 512)    0           ['max_pooling2d_3[0][0]']        
                                                                                                  
 conv2d_8 (Conv2D)              (None, 8, 8, 1024)   4719616     ['dropout_3[0][0]']              
                                                                                                  
 conv2d_9 (Conv2D)              (None, 8, 8, 1024)   9438208     ['conv2d_8[0][0]']               
                                                                                                  
 activation_4 (Activation)      (None, 8, 8, 1024)   0           ['conv2d_9[0][0]']               
                                                                                                  
 conv2d_transpose (Conv2DTransp  (None, 16, 16, 512)  4719104    ['activation_4[0][0]']           
 ose)                                                                                             
                                                                                                  
 concatenate (Concatenate)      (None, 16, 16, 1024  0           ['conv2d_transpose[0][0]',       
                                )                                 'activation_3[0][0]']           
                                                                                                  
 dropout_4 (Dropout)            (None, 16, 16, 1024  0           ['concatenate[0][0]']            
                                )                                                                 
                                                                                                  
 conv2d_10 (Conv2D)             (None, 16, 16, 512)  4719104     ['dropout_4[0][0]']              
                                                                                                  
 conv2d_11 (Conv2D)             (None, 16, 16, 512)  2359808     ['conv2d_10[0][0]']              
                                                                                                  
 activation_5 (Activation)      (None, 16, 16, 512)  0           ['conv2d_11[0][0]']              
                                                                                                  
 conv2d_transpose_1 (Conv2DTran  (None, 32, 32, 256)  1179904    ['activation_5[0][0]']           
 spose)                                                                                           
                                                                                                  
 concatenate_1 (Concatenate)    (None, 32, 32, 512)  0           ['conv2d_transpose_1[0][0]',     
                                                                  'activation_2[0][0]']           
                                                                                                  
 dropout_5 (Dropout)            (None, 32, 32, 512)  0           ['concatenate_1[0][0]']          
                                                                                                  
 conv2d_12 (Conv2D)             (None, 32, 32, 256)  1179904     ['dropout_5[0][0]']              
                                                                                                  
 conv2d_13 (Conv2D)             (None, 32, 32, 256)  590080      ['conv2d_12[0][0]']              
                                                                                                  
 activation_6 (Activation)      (None, 32, 32, 256)  0           ['conv2d_13[0][0]']              
                                                                                                  
 conv2d_transpose_2 (Conv2DTran  (None, 64, 64, 128)  295040     ['activation_6[0][0]']           
 spose)                                                                                           
                                                                                                  
 concatenate_2 (Concatenate)    (None, 64, 64, 256)  0           ['conv2d_transpose_2[0][0]',     
                                                                  'activation_1[0][0]']           
                                                                                                  
 dropout_6 (Dropout)            (None, 64, 64, 256)  0           ['concatenate_2[0][0]']          
                                                                                                  
 conv2d_14 (Conv2D)             (None, 64, 64, 128)  295040      ['dropout_6[0][0]']              
                                                                                                  
 conv2d_15 (Conv2D)             (None, 64, 64, 128)  147584      ['conv2d_14[0][0]']              
                                                                                                  
 activation_7 (Activation)      (None, 64, 64, 128)  0           ['conv2d_15[0][0]']              
                                                                                                  
 conv2d_transpose_3 (Conv2DTran  (None, 128, 128, 64  73792      ['activation_7[0][0]']           
 spose)                         )                                                                 
                                                                                                  
 concatenate_3 (Concatenate)    (None, 128, 128, 12  0           ['conv2d_transpose_3[0][0]',     
                                8)                                'activation[0][0]']             
                                                                                                  
 dropout_7 (Dropout)            (None, 128, 128, 12  0           ['concatenate_3[0][0]']          
                                8)                                                                
                                                                                                  
 conv2d_16 (Conv2D)             (None, 128, 128, 64  73792       ['dropout_7[0][0]']              
                                )                                                                 
                                                                                                  
 conv2d_17 (Conv2D)             (None, 128, 128, 64  36928       ['conv2d_16[0][0]']              
                                )                                                                 
                                                                                                  
 activation_8 (Activation)      (None, 128, 128, 64  0           ['conv2d_17[0][0]']              
                                )                                                                 
                                                                                                  
 conv2d_18 (Conv2D)             (None, 128, 128, 3)  195         ['activation_8[0][0]']           
                                                                                                  
==================================================================================================
Total params: 34,513,475
Trainable params: 34,513,475
Non-trainable params: 0
__________________________________________________________________________________________________
"""