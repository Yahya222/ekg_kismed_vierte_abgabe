import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from Preprocess import *
from wettbewerb import load_references
from python_speech_features import logfbank
import numpy as np
import scipy.io
import scipy.interpolate
import pandas as pd
from sklearn import model_selection
import pathos.multiprocessing as mp
import cv2


def fit_tolength(source, length):
    """
    Resize signal to target shape
        Args:
            source: 1D signal
            length: target length in samples
        Returns:
            target: 1D signal with target length
    """
    target = np.zeros([length])
    w_l = min(source.size, target.size)
    target[0:w_l] = source[0:w_l]
    return target


def random_resample(signal, upscale_factor=1):
    """
    Resample signal to random sample length
        Args:
            signal: 1D signal
        Return:
            signal: 1D signal with random sample length
    """
    length = len(signal)
    new_length = np.random.randint(
        low=int(length * 80 / 120), high=int(length * 80 / 60)
    )
    sq_sig = stretch_squeeze(signal, new_length)
    signal = fit_tolength(sq_sig, length)
    return signal


def zero_filter(input, threshold=2, depth=8):
    """
    This function set values to zero at random places of 1D signal
        Args:
            input: 1D signal
            threshold: Parameter
            depth: Parameter
        Returns:
            1D zero-filtered signal
    """
    shape = input.shape
    # compensate for lost length due to mask processing
    noise_shape = [shape[0] + depth]

    # Generate random noise
    noise = np.random.normal(0, 1, noise_shape)

    # Pick positions where the noise is above a certain threshold
    mask = np.greater(noise, threshold)

    # grow a neighbourhood of True values with at least length depth+1
    for d in range(depth):
        mask = np.logical_or(mask[:-1], mask[1:])
    output = np.where(mask, np.zeros(shape), input)
    return output


def stretch_squeeze(source, length):
    """
    Stretch or squeeze signal with the given length
        Args:
            source: 1D signal
            length: Parameter
        Returns:
            result: modified signal
    """
    target = np.zeros([1, length])
    interpol_obj = scipy.interpolate.interp1d(np.arange(source.size), source)
    grid = np.linspace(0, source.size - 1, target.size)
    result = interpol_obj(grid)
    return result


def _logMelFilterbank(signal, parse_param=(0.003, 0.00145, 15), augment="NO"):
    """
    Compute the log Mel-Filterbanks. Similar to stft, it produces a map of frequency over time.
    It is used to generate features for training
        Args:
            signal: 1D signal
            parse_param: parameters
            augment: parameter

        Returns:
            fbank: A numpy array of shape (600, nfilt) = (600,15)

    """
    wave = signal
    if augment == "zero_filter":
        wave = zero_filter(wave, threshold=2, depth=10)
    if augment == "random_resample":
        wave = random_resample(wave)
    fbank = logfbank(
        wave,
        samplerate=len(wave),
        winlen=float(parse_param[0]),
        winstep=float(parse_param[1]),
        nfilt=int(parse_param[2]),
        nfft=1024,
    )
    fbank = cv2.resize(fbank, (15, 600), interpolation=cv2.INTER_CUBIC)
    return fbank


def load_data_leads(path="../training/"):
    """
    Load data from path, split into train and validation sets and
    save them as numpy arrays. In future this should be replaced with vaex to reduce RAM usage
    """
    codes = {"A": 0, "N": 1, "O": 2, "~": 3}
    ecg_leads, ecg_labels, _, _ = load_references(path)
    ecg_labels = (pd.Series(ecg_labels)).map(codes).tolist()  # mapping

    # generate validation file
    ecg_leads, x_test, ecg_labels, y_test = model_selection.train_test_split(
        ecg_leads, ecg_labels, test_size=0.1, stratify=ecg_labels, shuffle=True
    )
    np.save("leads.npy", ecg_leads)
    np.save("labels.npy", ecg_labels)
    np.save("x_test.npy", x_test)
    np.save("y_test.npy", y_test)


def log_mel_list(x_train, y_train, augment="NO"):
    """
    Wrapper function: Makes list of tuples
    """
    return [
        (_logMelFilterbank(file_name, augment=augment), y_train[index])
        for index, file_name in enumerate(x_train)
    ]


def gen_train_val_deskriptor():
    """
    This functions denoises the signals and returns denoised und normalised signals as arrays
    """
    load_data_leads()  # only do if data is in path
    ecg_leads = np.load("leads.npy", allow_pickle=True)
    ecg_labels = np.load("labels.npy", allow_pickle=True)
    val_leads = np.load("x_test.npy", allow_pickle=True)
    val_labels = np.load("y_test.npy", allow_pickle=True)

    print(len(ecg_leads))
    print(len(ecg_labels))

    pp = PreprocessingData(300)
    ecg_leads = pp.denoise_data(ecg_leads)
    val_leads = pp.denoise_data(val_leads)

    x_train = ecg_leads
    y_train = ecg_labels
    x_val = val_leads
    y_val = val_labels

    return x_train, x_val, y_train, y_val


def augmentation(x_train, x_val, y_train, y_val):
    """
    Augmentation of training data with multiprocessing and saves data as numpy arrays
        Args:
            x_train,x_val,y_train,y_val: data

    """
    arguments = [
        (x_train, y_train, "random_resample"),
        (x_train, y_train, "zero_filter"),
        (x_train, y_train, "NO"),
    ]
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(log_mel_list, arguments)
    augmented_train_set = []
    for i in range(len(results)):
        augmented_train_set.extend(results[i])

    training_features = [sample[0] for sample in augmented_train_set]
    training_labels = [sample[1] for sample in augmented_train_set]

    val_set_features = np.array(
        [_logMelFilterbank(file_name, augment="NO") for file_name in x_val]
    )
    val_set_labels = np.array(y_val)

    np.save("augmented_set_mfcc", training_features)
    np.save("augmented_set_labels", training_labels)
    np.save("validate_set_mfcc", val_set_features)
    np.save("validate_set_labels", val_set_labels)

    return print("files saved and finished augmenting")
