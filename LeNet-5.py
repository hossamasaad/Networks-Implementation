from keras.models import Model
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Input

class LeNet5(Model):

    def __init__(self):
        super(LeNet5, self).__init__(name="LeNet-5")

        self.conv1a = Conv2D(filters=6, kernel_size=(5,5), strides=1, activation='tanh')  # 28 * 28 * 6
        self.avgpool1a = AveragePooling2D(pool_size=2, strides=2)                         # 14 * 14 * 6

        self.conv2b = Conv2D(filters=16, kernel_size=(5,5), strides=1, activation='tanh') # 10 * 10 * 16
        self.avgpool2b = AveragePooling2D(pool_size=2, strides=2)                         # 5 * 5 * 16

        self.flatten = Flatten()                                                          # 400

        self.dense1 = Dense(120, activation='tanh')                                       # 120
        self.dense2 = Dense( 84, activation='tanh')                                       # 84
        self.dense3 = Dense( 10, activation='softmax')                                    # 10

    
    def call(self, inputs):
        # Block 1
        x = self.conv1a(inputs)
        x = self.avgpool1a(x)
        
        # Block 2
        x = self.conv2b(x)
        x = self.avgpool2b(x)

        # FullyConnected Layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        # classifier
        output = self.dense3(x)

        return output

lenet5 = LeNet5()
lenet5.build(input_shape=(None, 32, 32, 1))

lenet5.summary()