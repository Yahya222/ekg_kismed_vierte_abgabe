# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""
from trainings_val_data_gen import *
from model_2D import train_model

if __name__ == "__main__":  # bei multiprocessing auf Windows notwendig
    """
    Zuerst werden die Daten aus PATH in arrays gespeichert. Anschließend wird der Trainingsdatensatz
    augmentiert. Hierfür wird die Frequenz verändert und Signalverlauf modifiziert, d.h., dass an einigen Stützstellen
    durch null ersetzt wird.
    Das Modell ist ein SE-RESNET mit 18 Layers mit Klassifikator.
    """
    x_train, x_val, y_train, y_val = gen_train_val_deskriptor()
    augmentation(x_train, x_val, y_train, y_val)
    train_model()
