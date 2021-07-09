import scipy.io
import numpy as np
import scipy
import scipy.signal
from python_speech_features import logfbank
from scipy import signal
import cv2


class PreprocessingData:
    def __init__(self, fs):
        self.fs = fs

    def trim_data(self, x, length=7.5):
        """
        Parameters
        ----------
        x : List of numpy arrays
            List consists of signals, loaded from path
        length: float
            length in seconds
        Returns
        -------
        x : List of numpy arrays
            Trimmed data to minimum length of signal data list
        """
        # min_length = min(x, key=lambda k: k.shape[0]).shape[0]
        min_length = int(length * self.fs)
        for i in range(len(x)):
            x[i] = x[i][0:min_length]
        return x

    def cut_data(self, x, length=2):
        """
        Parameters
        ----------
        x : List of numpy arrays
            List consists of signals, loaded from path
        length: float
            length in seconds
        Returns
        -------
        x : List of numpy arrays
            Trimmed data to minimum length of signal data list
        """
        # min_length = min(x, key=lambda k: k.shape[0]).shape[0]
        start_length = int(length * self.fs)
        for i in range(len(x)):
            x[i] = x[i][start_length:]
        return x

    def bandpass_filter(self):
        """
        Parameters
        ----------

        Returns
        -------
        b, a : float
            Parameters for bandpass filter in scipy package
        """
        fs = 300  # sampling rate (in Hz)
        freq_pass = np.array([4.0, 35.0]) / (fs / 2.0)
        freq_stop = np.array([0.5, 50.0]) / (fs / 2.0)
        gain_pass = 1
        gain_stop = 20
        filt_order, cut_freq = signal.buttord(
            freq_pass, freq_stop, gain_pass, gain_stop
        )
        b, a = signal.butter(filt_order, cut_freq, "bandpass")
        return b, a

    def running_mean(self, x_single, k):
        """
        Parameters
        ----------
        x_single : Numpy array
           Single data
        k : int
           Factor for the running mean, smoothing factor
        Returns
        -------
        array : numpy array
           Filtered array
        """

        cumsum = np.cumsum(np.insert(x_single, 0, 0))
        return (cumsum[k:] - cumsum[:-k]) / float(k)

    def denoise_data(self, x):
        """
        Parameters
        ----------
        x : List of numpy arrays
           List consists of signals, loaded from path
        Returns
        -------
        x : List of numpy arrays
           Denoised signal normalized
        """

        x_filt = list()
        for i in range(len(x)):
            b, a = self.bandpass_filter()
            x_n = scipy.signal.filtfilt(b, a, x[i], axis=0)
            x_n = x_n / ((max(x_n) - min(x_n)) / 2)
            x_filt.append(x_n)
        return x_filt

    def white_noise_adder(self, x, k):
        """
        Parameters
        ----------
        x : List of numpy arrays
           List consists of signals, should be normalized
        Returns
        -------
        x : List of numpy arrays
           Added white noise to signal
        """

        x_noised_list = []
        for i in range(len(x)):
            noise = np.random.normal(0, k, x[i].shape[0])
            x_noised_list.append(noise + x[i])
        return x_noised_list

    def create_split_snippets(self, x, labels, min_len, overlap, seconds):
        """
        Parameters
        ----------
        x : List of numpy arrays
            List consists of signals, loaded from path
        labels : list of str
            List of string labels, which is associated with x
        min_len : int
            It defines the minimum length of signal chunks. Should be smaller or equal to smallest sample of list x.
        overlap : float
            Overlap signal chunks in seconds generally 50% overlap is good
        seconds : float
            Signal chunk length in seconds, for example 4 seconds ( 4 seconds = 4 * 300 Hz = 1200 samples)
        Returns
        -------
        signal_list : list of numpy arrays
            Splitted signal
        signal_label_list : list of strings
            Labels associated to splitted signals, should have same length!
        """

        signal_list = []
        signal_label_list = []
        rate = self.fs
        for j in range(len(x)):
            sig = x[j]  # take the signal from the list
            # Split signal with overlap
            for i in range(0, len(sig), int((seconds - overlap) * rate)):
                split = sig[i : i + int(seconds * rate)]

                # Check if exceeds the end of signal
                if len(split) < int(min_len * rate):
                    break

                # Signal chunk too short and bigger than minimum length as redundant parameter
                if len(split) < int(rate * seconds):  #
                    # Zero padding to same length
                    split = np.hstack(
                        (split, np.zeros((int(rate * seconds) - len(split))))
                    )

                split = np.array(split)  # make sure it is a numpy array
                signal_list.append(split)  # append to the list
                signal_label_list.append(
                    labels[j]
                )  # append the associated labels for the same signal
        return signal_list, signal_label_list

    def manipulate(self, x, shift_max, shift_direction="both"):
        """
        Parameters
        ----------
        x : List of numpy arrays
           List consists of signals
        shift_max : int
           Shift in seconds
        shift_direction : String
           Right or Both are available
        Returns
        -------
        augmented_data_list: List of numpy arrays
           List of manipulated data
        """
        augmented_data_list = []
        sampling_rate = self.fs
        for i in range(len(x)):
            shift = np.random.randint(sampling_rate * shift_max)
            stretch = np.random.randint(8, 12) / 10
            if shift_direction == "right":
                shift = -shift
            elif shift_direction == "both":
                direction = np.random.randint(0, 2)
                if direction == 1:
                    shift = -shift
            augmented_data = np.roll(x[i], shift)
            augmented_data = augmented_data.astype(float)
            # augmented_data = librosa.effects.time_stretch(augmented_data, rate=stretch)
            augmented_data_list.append(augmented_data)
        return augmented_data_list

    def fft_data(self, x):
        """
        Parameters
        ----------
        x : List of numpy arrays
           List consists of signals
        Returns
        -------
        y_f: List of numpy arrays
           List of FFT data, which is normalized
        N: int
           Number of data points, it should be same for all signals
        """
        xf = list()
        yf = list()
        for i in range(len(x)):
            N = len(x[i])
            fft_data = np.fft.fft(x[i])
            fft_data = 2 * np.abs(fft_data) / N
            fft_data = fft_data[: N // 2]
            fft_data[0] = 0.5 * fft_data[0]
            x_freq = np.fft.fftfreq(N, 1 / self.fs)[: N // 2]
            xf.append(x_freq)
            yf.append(fft_data)
        N = yf[0].shape[0]
        return yf, N

    def _logMelFilterbank(self, signal, parse_param=(0.003, 0.00145, 15), augment="NO"):
        """
        Compute the log Mel-Filterbanks
        Returns a numpy array of shape (600, nfilt) = (600,15)
        """
        wave = signal
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
