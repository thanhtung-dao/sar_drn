from keras.models import Model
from keras.layers import Input, Add
from keras.layers.convolutional import Conv2D
from tensorflow.math import log
from keras import backend as K
from tensorflow.logging import set_verbosity, FATAL

set_verbosity(FATAL)


def get_red30_model():
    """
    Define the model architecture

    Return the model
    """

    inputs = Input(shape=(256, 256, 3))

    conv1 = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
    )(inputs)
    short_cut_conv1 = conv1

    conv2 = Conv2D(
        64, (3, 3), dilation_rate=2, padding="SAME", activation="relu"
    )(conv1)
    conv3 = Conv2D(
        64, (3, 3), dilation_rate=3, padding="SAME", activation="relu"
    )(conv2)

    conv4 = Add()([conv3, short_cut_conv1])
    conv4 = Conv2D(
        64, (3, 3), dilation_rate=4, padding="SAME", activation="relu"
    )(conv4)

    short_cut_conv4 = conv4
    conv5 = Conv2D(
        64, (3, 3), dilation_rate=3, padding="SAME", activation="relu"
    )(conv4)
    conv6 = Conv2D(
        64, (3, 3), dilation_rate=2, padding="SAME", activation="relu"
    )(conv5)

    conv7 = Add()([conv6, short_cut_conv4])
    conv7 = Conv2D(3, (3, 3), padding="SAME")(conv7)

    model = Model(inputs=inputs, outputs=conv7)
    return model


def PSNR(y_true, y_pred):
    """
    @param y_true: target value
    @param y_pred: predicted value
    """
    max_pixel = 0.5
    y_pred = K.clip(y_pred, -0.5, 0.5)
    return 10.0 * log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


if __name__ == "__main__":

    """
    For Debugging purposes
    """
    model = get_red30_model()
    print(model.summary())
