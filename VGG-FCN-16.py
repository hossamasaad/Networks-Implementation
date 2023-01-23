from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, Activation, Add, Input


class FCN16(Model):


    def __init__(self, n_classes):
        super(FCN16, self).__init__(name='FCN-16')

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
        # path 1
        self.convT1 = Conv2DTranspose(filters=n_classes, kernel_size=(4, 4), strides=2, use_bias=False)
        self.cropT1 = Cropping2D(cropping=(1, 1))

        # path 2
        self.conv1d = Conv2D(filters=n_classes, kernel_size=(1, 1), padding='same', activation='relu')

        self.add = Add()

        # output
        self.convT2 = Conv2DTranspose(filters=n_classes, kernel_size=(16, 16), strides=16, use_bias=False)
        self.cropT2 = Cropping2D(cropping=(1, 1))

        self.softmax = Activation("softmax")

    def call(self, inputs):
        # VGG16 Encoder Blocks

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
        pool4 = x

        # Block 5
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv5c(x)
        x = self.max_pool5(x)

        x = self.extra1(x)
        x = self.extra2(x)
        pool5 = x

        # Decoder
        path1 = self.convT1(pool5)
        path1 = self.cropT1(path1)

        path2 = self.conv1d(pool4)

        add = self.add([path1, path2])

        output = self.convT2(add)
        output = self.cropT2(output)
        output = self.softmax(output)

        return output


if __name__ == '__main__':
    fcn_16 = FCN16(12)
    fcn_16.build(input_shape=(None, 224, 224, 3))
    fcn_16.summary()


"""
Model: "FCN-16"
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
                                                                 
 conv2d_transpose_1 (Conv2DT  multiple                 36864     
 ranspose)                                                       
                                                                 
 cropping2d_1 (Cropping2D)   multiple                  0         
                                                                 
 activation (Activation)     multiple                  0         
                                                                 
=================================================================
Total params: 135,089,996
Trainable params: 135,089,996
Non-trainable params: 0
_________________________________________________________________
"""
