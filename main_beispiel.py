from wettbewerb import *
from predict import *
from subprocess import call

# load files and compute the features

ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="test_examples/")

# Make predictions and save the file
call(["python", "train.py"])
save_predictions(predict_labels(ecg_leads, fs, ecg_names, use_pretrained=True))
