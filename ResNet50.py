from keras.models import Model
from keras.layers import Layer
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Dense, MaxPooling2D, AveragePooling2D, Activation, Add

class IdentityBlock(Layer):

    def __init__(self, filters):
        super(IdentityBlock, self).__init__()

        self.conv1 = Conv2D(filters=filters[0], kernel_size=(1, 1), strides = 1, padding='valid')
        self.batch_norm1 = BatchNormalization()
        self.activation1 = Activation('relu')

        self.conv2 = Conv2D(filters=filters[1], kernel_size=(3, 3), strides = 1, padding='same')
        self.batch_norm2 = BatchNormalization()
        self.activation2 = Activation('relu')

        self.conv3 = Conv2D(filters=filters[2], kernel_size=(1, 1), strides = 1, padding='valid')
        self.batch_norm3 = BatchNormalization()

        self.add = Add()
        self.activation3 = Activation('relu')


    def call(self, inputs):
        # First block in Identity block
        x = self.conv1(inputs)
        x = self.batch_norm1(x) 
        x = self.activation1(x)

        # Second Block in identity block
        x = self.conv2(x)
        x = self.batch_norm2(x) 
        x = self.activation2(x)

        # Third Block in identity block
        x = self.conv3(x)
        x = self.batch_norm3(x)

        # Add input_x to the main path 
        x = self.add(inputs=[x, inputs])
        x = self.activation3(x)

        return x

class ConvBlock(Layer):

    def __init__(self, filters, stride=2):
        super(ConvBlock, self).__init__()

        self.conv1 = Conv2D(filters=filters[0], kernel_size=(1, 1), strides = stride, padding='valid')
        self.batch_norm1 = BatchNormalization()
        self.activation1 = Activation('relu')

        self.conv2 = Conv2D(filters=filters[1], kernel_size=(3, 3), strides = 1, padding='same')
        self.batch_norm2 = BatchNormalization()
        self.activation2 = Activation('relu')

        self.conv3 = Conv2D(filters=filters[2], kernel_size=(1, 1), strides = 1, padding='same')
        self.batch_norm3 = BatchNormalization()

        self.input_conv = Conv2D(filters=filters[2], kernel_size=(1, 1), strides = stride, padding='valid')
        self.batch_norm4 = BatchNormalization()

        self.add = Add()
        self.activation5 = Activation('relu')


    def call(self, inputs):
        # First block in Identity block
        x = self.conv1(inputs)
        x = self.batch_norm1(x) 
        x = self.activation1(x)

        # Second Block in identity block
        x = self.conv2(x)
        x = self.batch_norm2(x) 
        x = self.activation2(x)

        # Third Block in identity block
        x = self.conv3(x)
        x = self.batch_norm3(x)

        # shortcut path
        short_cut = self.input_conv(inputs)
        short_cut = self.batch_norm4(short_cut)

        # Add short_cut to the main path 
        x = self.add(inputs=[x, short_cut])
        x = self.activation5(x)

        return x

class ResNet50(Model):

    def __init__(self):
        super(ResNet50, self).__init__(name='ResNet50')

        self.conv       = Conv2D(filters=64, kernel_size=(7, 7), strides= 2)
        self.batch_norm = BatchNormalization()
        self.activation = Activation('relu')
        self.max_pool   = MaxPooling2D(pool_size=(3, 3), strides=2)

        self.conv_block2      = ConvBlock    (filters=[64, 64, 256], stride=1)
        self.identity_block2a = IdentityBlock(filters=[64, 64, 256])
        self.identity_block2b = IdentityBlock(filters=[64, 64, 256])

        self.conv_block3      = ConvBlock    (filters=[128, 128, 512])
        self.identity_block3a = IdentityBlock(filters=[128, 128, 512])
        self.identity_block3b = IdentityBlock(filters=[128, 128, 512])

        self.conv_block4      = ConvBlock    (filters=[256, 256, 1024])
        self.identity_block4a = IdentityBlock(filters=[256, 256, 1024])
        self.identity_block4b = IdentityBlock(filters=[256, 256, 1024])

        self.conv_block5      = ConvBlock    (filters=[512, 512, 2048])
        self.identity_block5a = IdentityBlock(filters=[512, 512, 2048])
        self.identity_block5b = IdentityBlock(filters=[512, 512, 2048])

        self.average_pool = AveragePooling2D(pool_size=(2, 2), strides= 1)
        self.flatten      = Flatten()
        self.dense        = Dense(6, activation='softmax')


    def call(self, inputs):
        print("\n\n\n\n")
        # Block-1
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        print(x)
        
        # Block-2
        x = self.conv_block2(x)
        x = self.identity_block2a(x)
        x = self.identity_block2b(x)
        print(x)

        # Block-3
        x = self.conv_block3(x)
        x = self.identity_block3a(x)
        x = self.identity_block3b(x)
        print(x)


        # Block-4
        x = self.conv_block4(x)
        x = self.identity_block4a(x)
        x = self.identity_block4b(x)
        print(x)


        # Block-5
        x = self.conv_block5    (x)
        x = self.identity_block5a(x)
        x = self.identity_block5b(x)
        print(x)

        # Pooling
        x = self.average_pool(x)
        print(x)

        # classifier
        x = self.flatten(x)
        x = self.dense(x)
        print(x)

        return x


resnet50 = ResNet50()
resnet50.build(input_shape=(None, 64, 64, 3))
resnet50.summary()