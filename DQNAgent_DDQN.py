import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import merge
import keras
from keras.optimizers import Adam, SGD
from keras.initializers import RandomUniform
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.initializers import he_uniform, glorot_uniform
from keras.layers.normalization import BatchNormalization
import random
import sqlite3
import time
import socket
import struct
import time
import sys
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from keras.utils.np_utils import to_categorical
import csv
from smallpt_pybind import *
import copy
from keras.callbacks import History
from keras.layers import Input, Dense, Lambda, Add
from keras import backend as K
from keras.models import Model
import tensorflow



class DQN:
    def __init__(self, params):
        self.state_space = params['state_space']
        self.action_space = params['action_space']
        self.learning_rate = params['learning_rate']
        self.dense_layer = params['dense_layer']
        self.state_layer = params['state_layer']
        self.advantage_layer = params['advantage_layer']
        self.exploration_rate = 1 # exploration rate
        self.epsilon_decay = 0.999
        self.epsilon_decay_linear = params['epsilon_decay_linear']
        self.epsilon_min = 0.05
        self.model = self.network()
        self.model_target = self.network_target()
        self.dict_act_dir = {}

    # Defining DDQN.
    # properties of DQN ------------------------------
    def network(self, weights=None):

        input_layer = Input(shape=(self.state_space,))
        x = (Dense(self.dense_layer, activation='relu'))(input_layer)
        x = (Dense(self.dense_layer, activation='relu'))(x)
        x = (Dense(self.dense_layer, activation='relu'))(x)

        xs = (Dense(self.state_layer, activation='relu'))(x)
        state_value = Dense(1)(xs)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.action_space,))(state_value)

        xa = (Dense(self.advantage_layer))(x)
        xa = (Dense(self.advantage_layer))(xa)
        action_advantage = Dense(self.action_space)(xa)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.action_space,))(action_advantage)


        X = Add()([state_value, action_advantage])
        model = Model(input=input_layer, output=X)
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
            print("weights loaded")
        return model

    # TARGET -------------------------------- !!!!!!!!
    def network_target(self, weights=None):
        input_layer = Input(shape=(self.state_space,))
        x = (Dense(self.dense_layer, activation='relu'))(input_layer)
        x = (Dense(self.dense_layer, activation='relu'))(x)
        x = (Dense(self.dense_layer, activation='relu'))(x)

        xs = (Dense(self.state_layer, activation='relu'))(x)
        state_value = Dense(1)(xs)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.action_space,))(state_value)

        xa = (Dense(self.advantage_layer))(x)
        xa = (Dense(self.advantage_layer))(xa)
        action_advantage = Dense(self.action_space)(xa)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                                  output_shape=(self.action_space,))(action_advantage)

        X = Add()([state_value, action_advantage])
        model = Model(input=input_layer, output=X)
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
            print("weights loaded")
        return model

    # Predict action based on current state
    def do_action(self, state, hitobj, dict_act):
        if np.random.rand() <= self.exploration_rate:
            action_idx = np.random.randint(0, self.action_space)
            hitobj.set_prob(1)        # microptimization
            hitobj.set_BRDF(1)        # microptimization
            hitobj.set_costheta(1)
            return action_idx
        arr = self.model.predict(state.reshape(1, self.state_space))
        arr[0] = arr[0].clip(min=0.1)

        idx_action = get_proportional_action(arr[0], len(arr[0]))
        if (idx_action < 12):
            prob_patch = 0.045620675
        elif (idx_action >= 12 and idx_action < 24):
            prob_patch = 0.050461491
        elif (idx_action >= 24 and idx_action < 36):
            prob_patch = 0.057276365
        elif (idx_action >= 36 and idx_action < 48):
            prob_patch = 0.067940351
        elif (idx_action >= 48 and idx_action < 60):
            prob_patch = 0.088541589
        else:
            prob_patch = 0.213758305

        hitobj.set_prob((arr[0].sum()*prob_patch)/(arr[0][idx_action]))  # prob of getting the hights value (1) times prob scattering to a specific direction inside the action patch
        hitobj.set_BRDF(1/math.pi)
        hitobj.set_costheta(dict_act[idx_action].get_z())
        return idx_action


    def train(self, state, action, reward, next_state, done,BRDF, dict_act, nl, params):
        if done == 1:
            target = reward
        else:
            prediction = self.model_target.predict(next_state.reshape(1, self.state_space))
            prediction[0] = prediction[0].clip(min=0.1)

            # Max Q
            if params['select_max_Q']:
                action_int_q = np.argmax(prediction[0])
                cos_theta_q = dict_act[action_int_q].get_z()
                target = reward + np.amax(prediction[0]) * cos_theta_q * BRDF
            else:
                # Average Q
                cumulative_q_value = cumulative_q(dict_act, nl, prediction[0])
                target = reward + 1 / self.action_space * cumulative_q_value * BRDF

        target_f = self.model.predict(state.reshape(1, self.state_space))
        target_f[0] = target_f[0].clip(min=0.1)
        target_f[0][action] = target
        self.model.fit(state.reshape(1, self.state_space), target_f, epochs=1, verbose=0)



