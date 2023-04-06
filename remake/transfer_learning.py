import tensorflow as tf
import os
import json
import time

import sketch_rnn_train
import model
import latent_space_explorer


DATA_DIRECTORY = ''
DATASET = 'sketchrnn-pig.npz'
MODEL_DIRECTORY = '/tmp/sketch_rnn/models/aaron_sheep/lstm'
SAVE_DIRECTORY = 'saved_model'


class DataTweaker:
    def __init__(self, datasets):
        (
            self.__train_set,
            self.__valid_set,
            self.__test_set
        ) = datasets

        self.__tweakData()
        self.__displayRandomSample()

    def getTrainSet(self, tweaked=False):
        if tweaked:
            return self.__tweaked_train_set
        else:
            return self.__train_set

    def getValidSet(self, tweaked=False):
        if tweaked:
            return self.__tweaked_valid_set
        else:
            return self.__valid_set

    def getTestSet(self, tweaked=False):
        if tweaked:
            return self.__tweaked_test_set
        else:
            return self.__test_set

    def __tweakData(self):
        self.__tweaked_train_set = self.__train_set
        self.__tweaked_valid_set = self.__valid_set
        self.__tweaked_test_set = self.__test_set

    def __displayRandomSample(self):
        latent_space_explorer.DataUtilities.strokes_to_svg(self.__train_set.random_sample())

if __name__ == '__main__':
    pass
