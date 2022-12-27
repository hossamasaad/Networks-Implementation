from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

class VGG16(Model):

    def __init__(self):
        super(VGG16, self).__init__(name='VGG-16')

        self.conv1a = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv1b = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv2a = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv2b = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv3a = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv3b = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv3c = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv4a = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv4b = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv4c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv5a = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv5b = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv5c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()

        self.dense1 = Dense(4096, activation='relu')
        self.dense2 = Dense(4096, activation='relu')
        self.dense3 = Dense(10  , activation='softmax')

    def call(self, inputs):

        # Block 1
        x = self.conv1a(inputs)
        x = self.conv1b(x)
        x = self.max_pool1(x)

        # Block 2
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.max_pool2(x)

        # Block 3
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.max_pool3(x)

        # Block 4
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.conv4c(x)
        x = self.max_pool4(x)


        # Block 5
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv5c(x)
        x = self.max_pool5(x)

        # Fully Connected Layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        # classifier
        output = self.dense3(x)

        return output

vgg16 = VGG16()
vgg16.build(input_shape=(None, 224, 224, 3))
vgg16.summary()

"""
Model: "VGG-16"
______________________________________________________________________
 Layer (type)                     Output Shape              Param #   
======================================================================
 conv2d (Conv2D)                  multiple                  1792      
                                                                 
 conv2d_1 (Conv2D)                multiple                  36928     
                                                                 
 max_pooling2d (MaxPooling2D)     multiple                  0                                                                        
                                                                 
 conv2d_2 (Conv2D)                multiple                  73856     
                                                                 
 conv2d_3 (Conv2D)                multiple                  147584    
                                                                 
 max_pooling2d_1 (MaxPooling2D)   multiple                  0                                        
                                                                 
 conv2d_4 (Conv2D)                multiple                  295168    
                                                                 
 conv2d_5 (Conv2D)                multiple                  590080    
                                                                 
 conv2d_6 (Conv2D)                multiple                  590080    
                                                                 
 max_pooling2d_2 (MaxPooling2D)   multiple                  0                                           
                                                                 
 conv2d_7 (Conv2D)                multiple                  1180160   
                                                                 
 conv2d_8 (Conv2D)                multiple                  2359808   
                                                                 
 conv2d_9 (Conv2D)                multiple                  2359808   
                                                                 
 max_pooling2d_3 (MaxPooling2D)   multiple                  0                                           
                                                                 
 conv2d_10 (Conv2D)               multiple                  2359808   
                                                                 
 conv2d_11 (Conv2D)               multiple                  2359808   
                                                                 
 conv2d_12 (Conv2D)               multiple                  2359808   
                                                                 
 max_pooling2d_4 (MaxPooling2D    multiple                  0                                                                   
                                                                 
 flatten (Flatten)                multiple                  0         
                                                                 
 dense (Dense)                    multiple                  102764544 
                                                                 
 dense_1 (Dense)                  multiple                  16781312  
                                                                 
 dense_2 (Dense)                  multiple                  40970     
                                                                 
======================================================================
Total params: 134,301,514
Trainable params: 134,301,514
Non-trainable params: 0
______________________________________________________________________
"""