from helper_layers import MaxUnpooling2D, MaxPoolingWithArgmax2D
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    Activation,
    BatchNormalization,
)


def conv_block(inputs, n_filters):
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def vgg_encoder(inputs):
    x = conv_block(inputs, 64)
    x = conv_block(x, 64)
    x, mask1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)

    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x, mask2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)

    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x, mask3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)

    x = conv_block(x, 512)
    x = conv_block(x, 512)
    x = conv_block(x, 512)
    x, mask4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)

    x = conv_block(x, 512)
    x = conv_block(x, 512)
    x = conv_block(x, 512)
    x, mask5 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)

    return x, (mask1, mask2, mask3, mask4, mask5)


def segnet_decoder(inputs, masks, n_classes):

    x = MaxUnpooling2D(size=(2, 2))([inputs, masks[4]])
    x = conv_block(x, 512)
    x = conv_block(x, 512)
    x = conv_block(x, 512)

    x = MaxUnpooling2D(size=(2, 2))([x, masks[3]])
    x = conv_block(x, 512)
    x = conv_block(x, 512)
    x = conv_block(x, 512)

    x = MaxUnpooling2D(size=(2, 2))([x, masks[2]])
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x = conv_block(x, 256)

    x = MaxUnpooling2D(size=(2, 2))([x, masks[1]])
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    x = MaxUnpooling2D(size=(2, 2))([x, masks[0]])
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    output = Conv2D(filters=n_classes, kernel_size=(1, 1), strides=1, activation='softmax')(x)

    return output


def SegNet(n_classes):
    inputs = Input(shape=(224, 224, 3,))
    x, masks = vgg_encoder(inputs)
    decoder = segnet_decoder(x, masks, n_classes)
    model = Model(inputs=inputs, outputs=decoder)

    return model


segnet = SegNet(32)
segnet.summary()