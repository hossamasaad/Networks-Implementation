from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input


class AlexNet(Model):

    def __init__(self):
        super(AlexNet, self).__init__(name='AlexNet')

        self.conv1a = Conv2D(filters=96, kernel_size=(11, 11), strides=4, activation='relu')
        self.batch_normalization1a = BatchNormalization()
        self.max_pool1a = MaxPooling2D(pool_size=(3, 3), strides=2)

        self.conv2a = Conv2D(filters=256, kernel_size=(5, 5), strides=1, activation='relu', padding='same')
        self.batch_normalization2a = BatchNormalization()
        self.max_pool2a = MaxPooling2D(pool_size=(3, 3), strides=2)

        self.conv3a = Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        self.batch_normalization3a = BatchNormalization()
        
        self.conv4a = Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        self.batch_normalization4a = BatchNormalization()
        
        self.conv5a = Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        self.batch_normalization5a = BatchNormalization()
        self.max_pool5a = MaxPooling2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()

        self.dense1 = Dense(4096, activation='relu')
        self.dropout1 = Dropout(rate=0.4)

        self.dense2 = Dense(4096, activation='relu')
        self.dropout2 = Dropout(rate=0.4)
        
        self.dense3 = Dense(1000, activation='softmax')

    def call(self, input):
        # Block 1
        x = self.conv1a(input)
        x = self.batch_normalization1a(x)
        x = self.max_pool1a(x)

        # Block 2
        x = self.conv2a(x)
        x = self.batch_normalization2a(x)
        x = self.max_pool2a(x)

        # Block 3
        x = self.conv3a(x)
        x = self.batch_normalization3a(x)
        
        # Block 4
        x = self.conv4a(x)
        x = self.batch_normalization4a(x)

        # Block 5
        x = self.conv5a(x)
        x = self.batch_normalization5a(x)
        x = self.max_pool5a(x)

        # FullyConnected Layers
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.dropout2(x)

        # Classifier
        output = self.dense3(x)

        return output

alexnet = AlexNet()
alexnet.build(input_shape=(None, 227, 227, 3))
alexnet.summary()

"""
Model: "AlexNet"
________________________________________________________________________________________________
 Layer (type)                                             Output Shape              Param #   
================================================================================================
 conv2d (Conv2D)                                            multiple                34944     
                                                                 
 batch_normalization (BatchNormalization)                   multiple                384                                                        
                                                                 
 max_pooling2d (MaxPooling2D)                               multiple                0         
                                                                    
 conv2d_1 (Conv2D)                                          multiple                614656    
                                                                 
 batch_normalization_1 (BatchNormalization)                 multiple                1024      
                                                                                                                  
 max_pooling2d_1 (MaxPooling2D)                             multiple                0         
                                                           
 conv2d_2 (Conv2D)                                          multiple                885120    
                                                                 
 batch_normalization_2 (BatchNormalization)                 multiple                1536      
                                                                                      
 conv2d_3 (Conv2D)                                          multiple                1327488   
                                                                 
 batch_normalization_3 (BatchNormalization)                 multiple                1536      
                                                                                      
 conv2d_4 (Conv2D)                                          multiple                1327488   
                                                                 
 batch_normalization_4 (BatchNormalization)                 multiple                1536      
                                                         
 max_pooling2d_2 (MaxPooling2D)                             multiple                0                                           
                                                                 
 flatten (Flatten)                                          multiple                0         
                                                                 
 dense (Dense)                                              multiple                56627200  
                                                                 
 dropout (Dropout)                                          multiple                0         
                                                                 
 dense_1 (Dense)                                            multiple                16781312  
                                                                 
 dropout_1 (Dropout)                                        multiple                0         
                                                                 
 dense_2 (Dense)                                            multiple                4097000   
                                                                 
===============================================================================================
Total params: 81,701,224
Trainable params: 81,698,216
Non-trainable params: 3,008
_______________________________________________________________________________________________
"""