from keras.models import Model
from keras.layers import Input, Add, PReLU, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout,concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from tensorflow.math import log
from keras import backend as K
from tensorflow.logging import set_verbosity, FATAL

# model architecture and loss function

set_verbosity(FATAL)

def get_red30_model():
    """
    Define the model architecture

    Return the model
    """

    inputs = Input(shape=(256,256, 3))

    enc_conv0 = Conv2D(64, (3, 3), padding="same",activation='relu', kernel_initializer="he_normal")(inputs)
    short_cut_enc_conv0 = enc_conv0

    enc_conv1 = Conv2D(64, (3, 3), dilation_rate = 2, padding="SAME", activation='relu')(enc_conv0)
    enc_conv2 = Conv2D(64, (3, 3), dilation_rate = 3, padding="SAME", activation='relu')(enc_conv1)

    enc_conv3 = Add()([enc_conv2, short_cut_enc_conv0])
    enc_conv3 = Conv2D(64, (3, 3), dilation_rate = 4, padding="SAME", activation='relu')(enc_conv3)

    short_cut_enc_conv3 = enc_conv3
    enc_conv4 = Conv2D(64, (3, 3), dilation_rate = 3, padding="SAME", activation='relu')(enc_conv3)
    enc_conv5 = Conv2D(64, (3, 3), dilation_rate = 2, padding="SAME", activation='relu')(enc_conv4)

    enc_conv6 = Add()([enc_conv5, short_cut_enc_conv3])
    enc_conv6 = Conv2D(3, (3, 3), padding='SAME')(enc_conv6)

    model = Model(inputs=inputs,outputs=enc_conv6)
    return model


def PSNR(y_true, y_pred):
    """
    @param y_true: target value
    @param y_pred: predicted value
    """
    max_pixel = 0.5
    y_pred = K.clip(y_pred, -0.5,0.5)
    return 10.0 * log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

if __name__ == '__main__':

    """
    For Debugging purposes
    """
    model = get_red30_model()
    print(model.summary())
