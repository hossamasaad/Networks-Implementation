from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Concatenate, AveragePooling2D, Flatten
from keras.models import Model
from keras.models import save_model


class InceptionBlock(Layer):

    def __init__(self, filters):
        super(InceptionBlock, self).__init__(name="InceptionBlock")

        # path-1
        self.conv1 = Conv2D(filters=filters[0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')

        # path-2
        self.conv2a = Conv2D(filters=filters[1][0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')
        self.conv2b = Conv2D(filters=filters[1][1], kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        
        # path-3
        self.conv3a = Conv2D(filters=filters[2][0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')
        self.conv3b = Conv2D(filters=filters[2][1], kernel_size=(5, 5), strides=1, padding='same', activation='relu')

        # path-4
        self.max_pool4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')
        self.conv4 = Conv2D(filters=filters[3], kernel_size=(1, 1), strides=1, padding='same', activation='relu')
        
        # Concatenate paths
        self.concat = Concatenate()

    def call(self, inputs):

        # path-1
        path_1 = self.conv1(inputs)

        # path-2
        path_2 = self.conv2a(inputs)
        path_2 = self.conv2b(path_2)

        # path-3
        path_3 = self.conv3a(inputs)
        path_3 = self.conv3b(path_3)

        # path-4
        path_4 = self.max_pool4(inputs)
        path_4 = self.conv4(path_4)

        # concatenate
        depth_concat = self.concat([path_1, path_2, path_3, path_4])

        return depth_concat


class GoogleNet(Model):
    def __init__(self):
        super(GoogleNet, self).__init__(name='GooglNet')

        # Block-1
        self.conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')
        self.max_pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

        # Block-2
        self.conv2a = Conv2D(filters= 64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')
        self.conv2b = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

        # Block-3: InceptionBlock
        self.inception_block3 = InceptionBlock(filters=[64, [96, 128], [16, 32], 32])

        # Block-4: InceptionBlock + MaxPoooling
        self.inception_block4 = InceptionBlock(filters=[128, [128, 192], [32, 96], 64])
        self.max_pool4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

        # Block-5: InceptionBlock
        self.inception_block5 = InceptionBlock(filters=[192, [96, 208], [16, 48], 64])

        # Extra-output1: AvgPooling + Conv + Flatten + Dense + Dense(Softmax)
        self.avg_pool_ex1 = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')
        self.conv_ex1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')
        self.flatten_ex1 = Flatten()
        self.dense_ex1a = Dense(units=1024, activation='relu')
        self.dense_ex1b = Dense(units=5   , activation='softmax')

        # Block-6
        self.inception_block6 = InceptionBlock(filters=[160, [112, 224], [24, 64], 64])

        # Block-7
        self.inception_block7 = InceptionBlock(filters=[128, [128, 256], [24, 64], 64])

        # Block-8
        self.inception_block8 = InceptionBlock(filters=[112, [144, 288], [32, 64], 64])

        # Extra-output2: AvgPooling + Conv + Flatten + Dense + Dense(Softmax)
        self.avg_pool_ex2 = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')
        self.conv_ex2 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')
        self.flatten_ex2 = Flatten()
        self.dense_ex2a = Dense(units=1024, activation='relu')
        self.dense_ex2b = Dense(units=1000, activation='softmax')
        
        # Block-9
        self.inception_block9 = InceptionBlock(filters=[256, [160, 320], [32, 128], 128])
        self.max_pool9 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

        # Block-10
        self.inception_block10 = InceptionBlock(filters=[256, [160, 320], [32, 128], 128])

        # Block-11
        self.inception_block11 = InceptionBlock(filters=[384, [192, 384], [48, 128], 128])

        # classifier
        self.average_pool = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')
        self.flatten = Flatten()
        self.dense = Dense(1000, activation='softmax')
    
    def call(self, inputs):

        # Block-1
        x = self.conv1(inputs)
        x = self.max_pool1(x)

        # Block-2
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.max_pool2(x)

        # Block-3
        x = self.inception_block3(x)

        # Block-4
        x = self.inception_block4(x)
        x = self.max_pool4(x)

        # Block-5
        x = self.inception_block5(x)
        
        # extra-output-1
        output1 = self.avg_pool_ex1(x)
        output1 = self.conv_ex1(output1)
        output1 = self.flatten_ex1(output1)
        output1 = self.dense_ex1a(output1)
        output1 = self.dense_ex1b(output1)

        # Block-6
        x = self.inception_block6(x)

        # Block-7
        x = self.inception_block7(x)

        # block-8
        x = self.inception_block8(x)

        # extra-output-2
        output2 = self.avg_pool_ex2(x)
        output2 = self.conv_ex2(output2)
        output2 = self.flatten_ex2(output2)
        output2 = self.dense_ex2a(output2)
        output2 = self.dense_ex2b(output2)

        # Block-9
        x = self.inception_block9(x)
        x = self.max_pool9(x)

        # Block-10
        x = self.inception_block10(x)

        # Block-10
        x = self.inception_block11(x)

        # Main-output classifier
        x = self.average_pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return [x, output2, output1]