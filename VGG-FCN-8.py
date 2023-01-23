from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, Activation, Add, Input


class FCN8(Model):
    def __init__(self, n_classes):
        super(FCN8, self).__init__(name='FCN-8')

        # Encoder
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

        self.extra1 = Conv2D(filters=4096, kernel_size=(7, 7), activation="relu", padding="same", name="conv6")
        self.extra2 = Conv2D(filters=4096, kernel_size=(1, 1), activation="relu", padding="same", name="conv7")

        # Decoder
        # path 1.1
        self.convT1 = Conv2DTranspose(filters=n_classes, kernel_size=(4, 4), strides=2, use_bias=False)
        self.cropT1 = Cropping2D(cropping=(1, 1))

        # path 1.2
        self.conv1d1 = Conv2D(filters=n_classes, kernel_size=(1, 1), padding='same', activation='relu')

        # ADD-1
        self.add1 = Add()

        # path 2.1
        self.convT2 = Conv2DTranspose(filters=n_classes, kernel_size=(4, 4), strides=2, use_bias=False)
        self.cropT2 = Cropping2D(cropping=(1, 1))

        # path 2.2
        self.conv1d2 = Conv2D(filters=n_classes, kernel_size=(1, 1), padding='same', activation='relu')

        # ADD-2
        self.add2 = Add()

        # output
        self.convT3 = Conv2DTranspose(filters=n_classes, kernel_size=(8, 8), strides=8, use_bias=False)
        self.cropT3 = Cropping2D(cropping=(1, 1))

        self.softmax = Activation("softmax")

    def call(self, inputs):
        ############ VGG16 Encoder #############
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
        pool3 = x

        # Block 4
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.conv4c(x)
        x = self.max_pool4(x)
        pool4 = x

        # Block 5
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv5c(x)
        x = self.max_pool5(x)

        x = self.extra1(x)
        x = self.extra2(x)

        pool5 = x

        ############ Decoder #############

        # Path 1.1
        path11 = self.convT1(pool5)
        path11 = self.cropT1(path11)

        # Path 1.2
        path12 = self.conv1d1(pool4)

        # Path 2.1 = path1.1 + path1.2
        path21 = self.add1([path11, path12])
        path21 = self.convT2(path21)
        path21 = self.cropT2(path21)

        # path 2.2
        path22 = self.conv1d2(pool3)

        # Add
        add = self.add2([path21, path22])

        # output
        output = self.convT3(add)
        output = self.cropT3(output)
        output = self.softmax(output)

        return output


if __name__ == '__main__':
    fcn_8 = FCN8(12)
    fcn_8.build(input_shape=(None, 224, 224, 3))
    fcn_8.summary()

"""
Model: "FCN-8"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             multiple                  1792      
                                                                 
 conv2d_1 (Conv2D)           multiple                  36928     
                                                                 
 max_pooling2d (MaxPooling2D  multiple                 0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           multiple                  73856     
                                                                 
 conv2d_3 (Conv2D)           multiple                  147584    
                                                                 
 max_pooling2d_1 (MaxPooling  multiple                 0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           multiple                  295168    
                                                                 
 conv2d_5 (Conv2D)           multiple                  590080    
                                                                 
 conv2d_6 (Conv2D)           multiple                  590080    
                                                                 
 max_pooling2d_2 (MaxPooling  multiple                 0         
 2D)                                                             
                                                                 
 conv2d_7 (Conv2D)           multiple                  1180160   
                                                                 
 conv2d_8 (Conv2D)           multiple                  2359808   
                                                                 
 conv2d_9 (Conv2D)           multiple                  2359808   
                                                                 
 max_pooling2d_3 (MaxPooling  multiple                 0         
 2D)                                                             
                                                                 
 conv2d_10 (Conv2D)          multiple                  2359808   
                                                                 
 conv2d_11 (Conv2D)          multiple                  2359808   
                                                                 
 conv2d_12 (Conv2D)          multiple                  2359808   
                                                                 
 max_pooling2d_4 (MaxPooling  multiple                 0         
 2D)                                                             
                                                                 
 conv6 (Conv2D)              multiple                  102764544 
                                                                 
 conv7 (Conv2D)              multiple                  16781312  
                                                                 
 conv2d_transpose (Conv2DTra  multiple                 786432    
 nspose)                                                         
                                                                 
 cropping2d (Cropping2D)     multiple                  0         
                                                                 
 conv2d_13 (Conv2D)          multiple                  6156      
                                                                 
 add (Add)                   multiple                  0         
                                                                 
 conv2d_transpose_1 (Conv2DT  multiple                 2304      
 ranspose)                                                       
                                                                 
 cropping2d_1 (Cropping2D)   multiple                  0         
                                                                 
 conv2d_14 (Conv2D)          multiple                  3084      
                                                                 
 add_1 (Add)                 multiple                  0         
                                                                 
 conv2d_transpose_2 (Conv2DT  multiple                 9216      
 ranspose)                                                       
                                                                 
 cropping2d_2 (Cropping2D)   multiple                  0         
                                                                 
 activation (Activation)     multiple                  0         
                                                                 
=================================================================
Total params: 135,067,736
Trainable params: 135,067,736
Non-trainable params: 0
_________________________________________________________________
"""
