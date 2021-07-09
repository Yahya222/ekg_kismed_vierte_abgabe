import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Preprocess import *
from trainings_val_data_gen import _logMelFilterbank
import pandas as pd
import tensorflow as tf
from ProposedCNN import *
from tensorflow.keras.models import load_model

# Hyperparameter
EPOCHS = 25
BATCH_SIZE = 128
num_classes = 4


def train_classifier_model():
    """
    This function loads pre trained model for two classes. The classifier layer is later replaced by
    4-class classification layer
    """
    source_model = load_model("model_last_epoch_2_class")
    # source model is pre-trained on two classes (A, N)
    model2 = Model(inputs=source_model.input, outputs=source_model.layers[-4].output)
    predictions = GlobalAveragePooling2D(name="Pooling")(model2.layers[-1].output)
    predictions = Dense(4, activation="softmax", name="Classificator")(predictions)
    model = Model(inputs=model2.input, outputs=predictions)

    model.summary()

    x_train = np.load("augmented_set_mfcc_2017.npy")
    y_train = np.load("augmented_set_labels_2017.npy")
    x_test = np.load("x_test.npy", allow_pickle=True)
    y_test = np.array(np.load("y_test.npy", allow_pickle=True))
    x_test = PreprocessingData(300).denoise_data(x_test)
    x_test = np.array(
        [_logMelFilterbank(file_name, augment="NO") for file_name in x_test]
    )
    print(x_train.shape)
    print(y_train.shape)

    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=4, verbose=1, min_delta=1e-4
    )
    opt = tf.keras.optimizers.Adam(learning_rate=1e-04)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        shuffle=True,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[reduce_lr_loss],
    )
    model.save("model_last_epoch_4_class")
