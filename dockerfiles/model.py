
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, ZeroPadding2D, Input, SeparableConv2D, Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D, GlobalAveragePooling2D, Reshape, Lambda, LSTM, concatenate, Activation, UpSampling2D, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.metrics import Recall, Precision, categorical_accuracy
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetV2S

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    true_sum = K.sum(K.square(y_true), -1)
    pred_sum = K.sum(K.square(y_pred), -1)
    return 1 - ((2. * intersection + smooth) / (true_sum + pred_sum + smooth))


def build_efficientNet():
    inputs = Input(shape=(None,None, 10), name="input_image")

    encoder = EfficientNetV2S(input_tensor=inputs, weights=None,include_top=False,include_preprocessing =False,classifier_activation=None)

    inp = encoder.input

    skip_connection_names = ["input_image", "block1b_project_activation","block2d_expand_activation","block4a_expand_activation", "block6a_expand_activation"]
    encoder_output = encoder.get_layer("top_activation").output

    f = [8,16,32, 64, 128,256]

    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)
        x = Activation("relu")(x)

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)
        x = Activation("relu")(x)

    x = Conv2D(2, (1, 1), padding="same")(x)
    output = Activation("sigmoid")(x)

    #model = Model(inputs, x)
    model = Model(inputs=[inp], outputs=[output], name="unet")

    return model
