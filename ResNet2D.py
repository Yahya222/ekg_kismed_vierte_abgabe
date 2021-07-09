from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import L2


def se_block(block_input, num_filters, ratio=4):  # Squeeze and excitation block

    """
    Args:
        block_input: input tensor to the squeeze and excitation block
        num_filters: no. of filters/channels in block_input
        ratio: a hyperparameter that denotes the ratio by which no. of channels will be reduced

    Returns:
        scale: scaled tensor after getting multiplied by new channel weights
    """

    pool1 = GlobalAveragePooling2D()(block_input)
    flat = Reshape((1, 1, num_filters))(pool1)
    dense1 = Dense(num_filters // ratio, activation="relu")(flat)
    dense2 = Dense(num_filters, activation="sigmoid")(dense1)
    scale = multiply([block_input, dense2])

    return scale


def resnet_block(block_input, num_filters):  # Single ResNet block

    """
    Args:
        block_input: input tensor to the ResNet block
        num_filters: no. of filters/channels in block_input

    Returns:
        relu2: activated tensor after addition with original input
    """
    if int_shape(block_input)[3] != num_filters:
        block_input = Conv2D(num_filters, kernel_size=(1, 1))(block_input)

    conv1 = Conv2D(num_filters, kernel_size=(3, 3), padding="same")(block_input)
    norm1 = BatchNormalization()(conv1)
    relu1 = Activation("relu")(norm1)
    conv2 = Conv2D(num_filters, kernel_size=(3, 3), padding="same")(relu1)
    norm2 = BatchNormalization()(conv2)

    se = se_block(norm2, num_filters=num_filters)

    sum = Add()([block_input, se])
    relu2 = Activation("relu")(sum)

    return relu2


def se_resnet14(input_shape):
    """
    Squeeze and excitation blocks applied on ResNet18. In additon MaxPooling is been used to
    reduce the dimensionality of input. This reduces the calculation time.
    Input size is 600x15x1 representing mel-spectograms of ecg signals.
    Output size is 4 representing classes to which ecg signals belong (softmax probability).
    """
    input = Input(shape=input_shape)
    input_rs = Reshape(input_shape=input_shape, target_shape=(600, 15, 1))(input)

    conv1 = Conv2D(16, kernel_size=(8, 8), activation="relu", padding="same")(input_rs)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 1))(conv1)

    block1 = resnet_block(conv1, 32)
    block2 = resnet_block(block1, 32)
    block2 = MaxPooling2D(pool_size=(2, 1))(block2)

    block3 = resnet_block(block2, 64)
    block4 = resnet_block(block3, 64)
    block4 = MaxPooling2D(pool_size=(2, 1))(block4)

    block5 = resnet_block(block4, 128)
    block6 = resnet_block(block5, 128)
    block6 = MaxPooling2D(pool_size=(2, 1))(block6)

    block7 = resnet_block(block6, 256)
    block8 = resnet_block(block7, 256)
    block8 = MaxPooling2D(pool_size=(2, 1))(block8)

    flat = BatchNormalization()(block8)
    flat = ReLU()(flat)
    flat = GlobalAveragePooling2D(name="feature_layer")(flat)
    flat = Dropout(0.2)(flat)
    output = Dense(4, activation="softmax")(flat)

    model = Model(inputs=input, outputs=output)
    return model
