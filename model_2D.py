import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Preprocess import *
from trainings_val_data_gen import _logMelFilterbank
import pandas as pd
import tensorflow as tf
from ResNet2D import se_resnet14
from tensorflow.keras.models import load_model


# Hyperparameter
EPOCHS = 100
BATCH_SIZE = 128
num_classes = 4

# Create SE-Resnet model with 18 Layers
def create_model(input_shape=(600, 15)):
    return se_resnet14(input_shape)


def train_model():
    """
    Train SE-RESNET with Adam optimizer.

    """
    # Load data 'A' == 0, 'N' ==  1,  'O'== 2, '~'== 3
    x_train = np.load("augmented_set_mfcc.npy")
    y_train = np.load("augmented_set_labels.npy")
    x_val = np.load("validate_set_mfcc.npy")
    y_val = np.load("validate_set_labels.npy")

    model = create_model()

    # set up train
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        "./model_trained/",
        save_best_only=True,
        monitor="val_loss",
        verbose=1,
        save_weights_only=False,
    )
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=6, verbose=1, min_delta=1e-4
    )
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
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
        validation_data=(x_val, y_val),
        shuffle=True,
        epochs=EPOCHS,
        verbose=2,
        callbacks=[mcp_save, reduce_lr_loss],
    )
    # save model
    # model.save('model_trained')
    return model


def predict_model(model, test_leads, test_names, fs):
    """
    Predict data set with the model.
        Args:
            model: trained model
            test_leads: list of signals
            test_names: list of names associated to signals
            fs: samplerate
        Returns:
            predictions: list of tuples with names and predicted labels
    """
    codes = {0: "A", 1: "N", 2: "O", 3: "~"}
    # evaluate the model
    predictions = []
    predicted_labels = []
    pp = PreprocessingData(fs)

    # denoise and create features
    test_leads = pp.denoise_data(test_leads)

    test_set_features = np.array(
        [_logMelFilterbank(file_name, augment="NO") for file_name in test_leads]
    )

    predicted_labels_raw = model.predict(test_set_features)

    [
        predicted_labels.append(np.argmax(elements))
        for elements in predicted_labels_raw.tolist()
    ]
    decoded_labels = (pd.Series(predicted_labels)).map(codes).tolist()

    predictions = list(zip(test_names, decoded_labels))

    return predictions


def eval_test():
    """
    For debugging, test the accuaracy of the model on test data set
    """
    model = load_model("./model_trained/")
    x_test = np.load("x_test.npy", allow_pickle=True)
    y_test = np.load("y_test.npy", allow_pickle=True)
    pp = PreprocessingData(300)
    # denoise and create features
    test_leads = pp.denoise_data(x_test)
    features = np.array(
        [_logMelFilterbank(file_name, augment="NO") for file_name in test_leads]
    )
    model.evaluate(features, y_test, verbose=1)
    return 0


def confusion_matrix(x_val, y_val):
    """
    Prints out the confusion matrix
    """
    model = load_model("./model_trained/")
    out = model.predict(x_val, verbose=1)
    y_pred = np.argmax(out, axis=1)
    y_true = y_val
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    print("Accuracy: {:.4f}".format(accuracy_score(y_pred, y_true)))
    print("Precision: {:.4f}".format(precision_score(y_pred, y_true, average="macro")))
    print("Recall: {:.4f}".format(recall_score(y_pred, y_true, average="macro")))
    print("F1 score: {:.4f}".format(f1_score(y_pred, y_true, average="macro")))
    cm = confusion_matrix(y_pred, y_true)
    print(cm)
