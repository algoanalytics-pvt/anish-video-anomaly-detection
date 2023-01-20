# external imports
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    UpSampling2D,
    Concatenate,
)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.activations import relu, sigmoid
from keras.optimizers import Adam
from keras.initializers import RandomNormal, Zeros
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import save_img, load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from os import listdir
from time import time
from sklearn import metrics
import json

# internal imports
import config
import utils


class sUAE:
    def __init__(
        self,
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        input_shape=config.image_size,
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_shape = (input_shape, input_shape, input_channels)
        self.create_model()

    def create_model(self):
        # initializers
        conv_weights_initializer = RandomNormal(mean=0.0, stddev=0.02)
        batch_norm_weights_initializer = RandomNormal(mean=1.0, stddev=0.02)
        batch_norm_bias_initializer = Zeros()
        # input
        input = Input(self.input_shape)
        x = Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(input)
        # x = Conv2D(16, kernel_size=(3, 3), padding="same", kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2D_1")(input)
        x = BatchNormalization()(x)
        x = relu(x)
        # encoder
        # downsize 1
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # x = Conv2D(32, kernel_size=(3, 3), padding="same", kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2D_2")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 2
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # x = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2D_3")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 3
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2D_4")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2), name=self.name + "_MaxPooling_4")(x)
            x = Conv2D(
                256,
                kernel_size=(3, 3),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            # x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2D_4")(x)
            x = BatchNormalization(name=self.name + "_BatchNormalization_5")(x)
            x = relu(x)
        # downsize 5
        if config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2), name=self.name + "_MaxPooling_5")(x)
            x = Conv2D(
                512,
                kernel_size=(3, 3),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            # x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2D_4")(x)
            x = BatchNormalization(name=self.name + "_BatchNormalization_6")(x)
            x = relu(x)
        # decoder
        # upsize 1
        x = Conv2DTranspose(
            128,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        # x = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2DTranspose_1")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 2
        x = Conv2DTranspose(
            64,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        # x = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2DTranspose_2")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 3
        x = Conv2DTranspose(
            32,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        # x = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2DTranspose_3")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
                name=self.name + "_Conv2DTranspose_4",
            )(x)
            # x = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2DTranspose_3")(x)
            x = BatchNormalization(name=self.name + "_BatchNormalization_10")(x)
            x = relu(x)
        # upsize 5
        if config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
                name=self.name + "_Conv2DTranspose_5",
            )(x)
            # x = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=conv_weights_initializer, name=self.name+"_Conv2DTranspose_3")(x)
            x = BatchNormalization(name=self.name + "_BatchNormalization_11")(x)
            x = relu(x)
        # output
        x = Conv2D(
            3,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # sigmoid
        x = sigmoid(x)
        # create the model
        self.model = Model(input, x)
        # initilalize Adam optimizer with configured learning rate
        optimizer_adam = Adam(learning_rate=config.learning_rate)
        # compile the model
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.model.compile(loss=utils.anomaly_loss, optimizer=optimizer_adam)

    def create_model_skip(self):
        # initializers
        conv_weights_initializer = RandomNormal(mean=0.0, stddev=0.02)
        batch_norm_weights_initializer = RandomNormal(mean=1.0, stddev=0.02)
        batch_norm_bias_initializer = Zeros()
        # input
        model_input = Input(self.input_shape)
        input_conv = Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(model_input)
        input_batch_norm = BatchNormalization()(input_conv)
        input_relu = relu(input_batch_norm)
        # encoder
        # downsize 1
        down_1_max_pool = MaxPooling2D(pool_size=(2, 2))(input_relu)
        down_1_conv = Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(down_1_max_pool)
        down_1_batch_norm = BatchNormalization()(down_1_conv)
        down_1_relu = relu(down_1_batch_norm)
        # downsize 2
        down_2_max_pool = MaxPooling2D(pool_size=(2, 2))(down_1_relu)
        down_2_conv = Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(down_2_max_pool)
        down_2_batch_norm = BatchNormalization()(down_2_conv)
        down_2_relu = relu(down_2_batch_norm)
        # downsize 3
        down_3_max_pool = MaxPooling2D(pool_size=(2, 2))(down_2_relu)
        down_3_conv = Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(down_3_max_pool)
        down_3_batch_norm = BatchNormalization()(down_3_conv)
        down_3_relu = relu(down_3_batch_norm)
        # decoder
        # upsize 1
        x = Conv2DTranspose(
            128,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(down_3_relu)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Concatenate(axis=-1)([down_2_relu, x])
        # upsize 2
        x = Conv2DTranspose(
            64,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 3
        x = Conv2DTranspose(
            32,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # output
        x = Conv2D(
            3,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # sigmoid
        x = sigmoid(x)
        # create the model
        self.model = Model(model_input, x)
        # initilalize Adam optimizer with configured learning rate
        optimizer_adam = Adam(learning_rate=config.learning_rate)
        # compile the model
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.model.compile(loss=utils.anomaly_loss, optimizer=optimizer_adam)

    def create_model_uae(self):
        input = Input(self.input_shape)
        x = Conv2D(64, kernel_size=(3, 3), padding="same")(input)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # down 1
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(128, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # down 2
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(256, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(256, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # down 3
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(512, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(512, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # up 1
        x = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(256, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(256, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # up 2
        x = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(128, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(128, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # up 3
        x = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # out
        x = Conv2D(3, kernel_size=(3, 3), padding="same")(x)
        # sigmoid
        x = sigmoid(x)
        # create the model
        self.model = Model(input, x)
        # initilalize Adam optimizer with configured learning rate
        optimizer_adam = Adam(learning_rate=config.learning_rate)
        # compile the model
        self.model.compile(loss=utils.anomaly_loss, optimizer=optimizer_adam)

    def create_model_straight_conv(self):
        # initializers
        conv_weights_initializer = RandomNormal(mean=0.0, stddev=0.02)
        batch_norm_weights_initializer = RandomNormal(mean=1.0, stddev=0.02)
        batch_norm_bias_initializer = Zeros()
        # input
        input = Input(self.input_shape)
        x = Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(input)
        x = BatchNormalization()(x)
        x = relu(x)
        # encoder
        # downsize 1
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 2
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 3
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # downsize 5
        if config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # decoder
        # upsize 1
        # x = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=conv_weights_initializer)(x)
        # x = BatchNormalization()(x)
        # x = relu(x)
        # upsize 2
        # x = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=conv_weights_initializer)(x)
        # x = BatchNormalization()(x)
        # x = relu(x)
        # upsize 3
        # x = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=conv_weights_initializer)(x)
        # x = BatchNormalization()(x)
        # x = relu(x)
        # upsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # upsize 5
        if config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(3, 3),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # output
        x = Conv2D(
            3,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # sigmoid
        x = sigmoid(x)
        # create the model
        self.model = Model(input, x)
        # initilalize Adam optimizer with configured learning rate
        optimizer_adam = Adam(learning_rate=config.learning_rate)
        # compile the model
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.model.compile(loss=utils.anomaly_loss, optimizer=optimizer_adam)

    def create_model_2_blocks(self):
        # initializers
        conv_weights_initializer = RandomNormal(mean=0.0, stddev=0.02)
        batch_norm_weights_initializer = RandomNormal(mean=1.0, stddev=0.02)
        batch_norm_bias_initializer = Zeros()
        # input
        input = Input(self.input_shape)
        x = Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(input)
        x = BatchNormalization()(x)
        x = relu(x)
        # encoder
        # downsize 1
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 2
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 3
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=conv_weights_initializer)(x)
        # x = BatchNormalization()(x)
        # x = relu(x)
        # downsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # downsize 5
        if config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # decoder
        # upsize 1
        x = Conv2DTranspose(
            256,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 2
        x = Conv2DTranspose(
            128,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 3
        # x = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=conv_weights_initializer)(x)
        # x = BatchNormalization()(x)
        # x = relu(x)
        # upsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # upsize 5
        if config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(3, 3),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # output
        x = Conv2D(
            3,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # sigmoid
        x = sigmoid(x)
        # create the model
        self.model = Model(input, x)
        # initilalize Adam optimizer with configured learning rate
        optimizer_adam = Adam(learning_rate=config.learning_rate)
        # compile the model
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.model.compile(loss=utils.anomaly_loss, optimizer=optimizer_adam)

    def create_model_2M(self):
        # initializers
        conv_weights_initializer = RandomNormal(mean=0.0, stddev=0.02)
        batch_norm_weights_initializer = RandomNormal(mean=1.0, stddev=0.02)
        batch_norm_bias_initializer = Zeros()
        # input
        input = Input(self.input_shape)
        x = Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(input)
        x = BatchNormalization()(x)
        x = relu(x)
        # encoder
        # downsize 1
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 2
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 3
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            512,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # downsize 5
        if config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # decoder
        # upsize 1
        x = Conv2DTranspose(
            512,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 2
        x = Conv2DTranspose(
            256,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 3
        x = Conv2DTranspose(
            128,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # upsize 5
        if config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(3, 3),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # output
        x = Conv2D(
            3,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # sigmoid
        x = sigmoid(x)
        # create the model
        self.model = Model(input, x)
        # initilalize Adam optimizer with configured learning rate
        optimizer_adam = Adam(learning_rate=config.learning_rate)
        # compile the model
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.model.compile(loss=utils.anomaly_loss, optimizer=optimizer_adam)

    def create_model_2m(self):
        # initializers
        conv_weights_initializer = RandomNormal(mean=0.0, stddev=0.02)
        batch_norm_weights_initializer = RandomNormal(mean=1.0, stddev=0.02)
        batch_norm_bias_initializer = Zeros()
        # input
        input = Input(self.input_shape)
        x = Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(input)
        x = BatchNormalization()(x)
        x = relu(x)
        # encoder
        # downsize 1
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 2
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 3
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            512,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # downsize 5
        if config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # decoder
        # upsize 1
        x = Conv2DTranspose(
            256,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 2
        x = Conv2DTranspose(
            128,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 3
        x = Conv2DTranspose(
            64,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # upsize 5
        if config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(3, 3),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # output
        x = Conv2D(
            3,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # sigmoid
        x = sigmoid(x)
        # create the model
        self.model = Model(input, x)
        # initilalize Adam optimizer with configured learning rate
        optimizer_adam = Adam(learning_rate=config.learning_rate)
        # compile the model
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.model.compile(loss=utils.anomaly_loss, optimizer=optimizer_adam)

    def create_model_sUAE_3M(self):
        # initializers
        conv_weights_initializer = RandomNormal(mean=0.0, stddev=0.02)
        batch_norm_weights_initializer = RandomNormal(mean=1.0, stddev=0.02)
        batch_norm_bias_initializer = Zeros()
        # input
        input = Input(self.input_shape)
        x = Conv2D(
            64,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(input)
        x = BatchNormalization()(x)
        x = relu(x)
        # encoder
        # downsize 1
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            128,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 2
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            256,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 3
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            1024,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # downsize 5
        if config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # decoder
        # upsize 1
        x = Conv2DTranspose(
            512,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 2
        x = Conv2DTranspose(
            256,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 3
        x = Conv2DTranspose(
            128,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # upsize 5
        if config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # output
        x = Conv2D(
            3,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # sigmoid
        x = sigmoid(x)
        # create the model
        self.model = Model(input, x)
        # initilalize Adam optimizer with configured learning rate
        optimizer_adam = Adam(learning_rate=config.learning_rate)
        # compile the model
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.model.compile(loss=utils.anomaly_loss, optimizer=optimizer_adam)

    def create_model_old(self):
        # initializers
        conv_weights_initializer = RandomNormal(mean=0.0, stddev=0.02)
        batch_norm_weights_initializer = RandomNormal(mean=1.0, stddev=0.02)
        batch_norm_bias_initializer = Zeros()
        # input
        input = Input(self.input_shape)
        x = Conv2D(
            64,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(input)
        x = BatchNormalization()(x)
        x = relu(x)
        # encoder
        # downsize 1
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            128,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 2
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            256,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 3
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            512,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # downsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # downsize 5
        if config.bottleneck_size == 7:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(
                256,
                kernel_size=(2, 2),
                padding="same",
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # decoder
        # upsize 1
        x = Conv2DTranspose(
            256,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 2
        x = Conv2DTranspose(
            128,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 3
        x = Conv2DTranspose(
            64,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer=conv_weights_initializer,
        )(x)
        x = BatchNormalization()(x)
        x = relu(x)
        # upsize 4
        if config.bottleneck_size == 14 or config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # upsize 5
        if config.bottleneck_size == 7:
            x = Conv2DTranspose(
                32,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_initializer=conv_weights_initializer,
            )(x)
            x = BatchNormalization()(x)
            x = relu(x)
        # output
        x = Conv2D(
            3,
            kernel_size=(2, 2),
            padding="same",
            kernel_initializer=conv_weights_initializer,
        )(x)
        # sigmoid
        x = sigmoid(x)
        # create the model
        self.model = Model(input, x)
        # initilalize Adam optimizer with configured learning rate
        optimizer_adam = Adam(learning_rate=config.learning_rate)
        # compile the model
        self.optimizer = Adam(learning_rate=config.learning_rate)
        self.model.compile(loss=utils.anomaly_loss, optimizer=optimizer_adam)

    def model_summary(self):
        self.model.summary(line_length=config.summary_line_length)

    def train(self):
        early_stopping = EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=config.patience
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            verbose=1,
            factor=0.2,
            patience=2,
            min_lr=0.000001,
        )
        model_checkpoint = ModelCheckpoint(
            config.weights_location,
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            min_delta=0,
        )
        callbacks = [reduce_lr, model_checkpoint, early_stopping]
        history = self.model.fit(
            utils.fixed_generator(utils.train_generator),
            epochs=config.epochs,
            steps_per_epoch=len(utils.train_generator),
            validation_data=utils.fixed_generator(utils.validation_generator),
            validation_steps=len(utils.validation_generator),
            # validation_split = 0.5,
            callbacks=callbacks,
        )
        hist = pd.DataFrame(history.history)
        hist["epoch"] = history.epoch
        plt.plot(hist["epoch"], hist["loss"], hist["val_loss"])
        plt.legend(["Training", "Validation"])
        plt.suptitle("UAE Training Plot", size=15)
        plt.title("", size=10)
        plt.ylabel("Anomaly Loss")
        plt.xlabel("Epochs")
        plt.savefig("")

    def save(self):
        self.model.save_weights(config.weights_location)
        self.model.save(config.model_location)
        converter = tf.lite.TFLiteConverter.from_saved_model(config.model_location)
        tflite_model = converter.convert()
        with open(config.tflite_model_location, "wb") as f:
            f.write(tflite_model)

    def load(self):
        self.model.load_weights(config.weights_location)

    def evaluate_training_losses(self):
        # Process Training frames
        counter = 0
        training_losses = []
        training_frames_folder = ""
        for image in natsorted(listdir(training_frames_folder)):
            counter = counter + 1
            print("Processing frame: " + str(counter))
            image = load_img(
                training_frames_folder + image, target_size=config.image_dimensions
            )
            array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)
            prediction = self.model.predict(array)
            loss = utils.anomaly_loss(array, prediction)
            training_losses.append(loss.numpy())
        np_training_losses = np.array(training_losses)
        print("Training Frames Statistics: ")
        print("Minimum Loss: ", np_training_losses.min())
        print("Maximum Loss: ", np_training_losses.max())
        print("Mean Loss: ", np_training_losses.mean())
        print("Standard Deviation: ", np_training_losses.std())

    def evaluate_validation_losses(self):
        # Process Validation frames
        validation_losses = []
        validation_frames_folder = ""
        for image in natsorted(listdir(validation_frames_folder)):
            print("Processing frame: " + image)
            image = load_img(
                validation_frames_folder + image, target_size=config.image_dimensions
            )
            array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)
            prediction = self.model.predict(array)
            loss = utils.anomaly_loss(array, prediction)
            validation_losses.append(loss.numpy())
        np_validation_losses = np.array(validation_losses)
        print("Validation Frames Statistics: ")
        print("Minimum Loss: ", np_validation_losses.min())
        print("Maximum Loss: ", np_validation_losses.max())
        print("Mean Loss: ", np_validation_losses.mean())
        print("Standard Deviation: ", np_validation_losses.std())

    def evaluate_testing_losses(self):
        # Process Testing frames
        testing_losses = []
        testing_frames_folder = ""
        for image in natsorted(listdir(testing_frames_folder)):
            print("Processing frame: " + image)
            image = load_img(
                testing_frames_folder + image, target_size=config.image_dimensions
            )
            array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)
            prediction = self.model.predict(array)
            loss = utils.anomaly_loss(array, prediction)
            testing_losses.append(loss.numpy())
        np_testing_losses = np.array(testing_losses)
        print("Testing Frames Statistics: ")
        print("Minimum Loss: ", np_testing_losses.min())
        print("Maximum Loss: ", np_testing_losses.max())
        print("Mean Loss: ", np_testing_losses.mean())
        print("Standard Deviation: ", np_testing_losses.std())

    def generate_reconsrtuction(
        self, input_filename, input_reshaped_filename, output_filename
    ):
        image = load_img(
            input_filename,
            color_mode=config.color_mode,
            target_size=config.image_dimensions,
        )
        # print(image.shape)
        # save_img(input_reshaped_filename, image)
        array = np.expand_dims(np.array(image) / 255.0, axis=[0]).astype(np.float32)
        print(array.shape)
        prediction = self.model.predict(array)
        print(utils.anomaly_loss(array, prediction).numpy())
        save_img(output_filename, prediction[0])

    def generate_losses(self):
        # save losses for normal frames
        normal_losses = []
        for image in natsorted(listdir(config.test_dataset_normal_folder)):
            print("Predicting frame: " + image)
            start = time()
            image = load_img(
                config.test_dataset_normal_folder + image,
                target_size=config.image_dimensions,
            )
            array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)
            prediction = self.model.predict(array)
            loss = utils.anomaly_loss(array, prediction)
            end = time()
            print("Time required: ", end - start)
            normal_losses.append(loss.numpy())
        self.normal_losses_count = len(normal_losses)
        normal_losses = np.array(normal_losses)
        np.save(config.normal_losses_save_path, normal_losses)
        # save losses for anomalous frames
        anomalous_losses = []
        for image in natsorted(listdir(config.test_dataset_anomalous_folder)):
            print("Predicting frame: " + image)
            start = time()
            image = load_img(
                config.test_dataset_anomalous_folder + image,
                target_size=config.image_dimensions,
            )
            array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)
            prediction = self.model.predict(array)
            loss = utils.anomaly_loss(array, prediction)
            end = time()
            print("Time required: ", end - start)
            anomalous_losses.append(loss.numpy())
        self.anomalous_losses_count = len(anomalous_losses)
        anomalous_losses = np.array(anomalous_losses)
        np.save(config.anomalous_losses_save_path, anomalous_losses)

    def generate_losses_sequential(self):
        losses = []
        for image_name in natsorted(listdir(config.test_dataset_sequential_folder)):
            print("Predicting frame: " + image_name)
            image = load_img(
                config.test_dataset_sequential_folder + image_name,
                target_size=config.image_dimensions,
            )
            array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)
            prediction = self.model.predict(array)
            loss = utils.anomaly_loss(array, prediction)
            losses.append(loss)
        losses_array = np.array(losses)
        np.save(config.losses_path, losses)

    def read_json_labels(self):
        labels_file = open(config.labels_path)
        labels_data = json.load(labels_file)
        count = labels_data["count"]
        labels = np.zeros((count,), dtype=np.int8)
        for event in labels_data["anomalous_frames"]:
            start = event["start"]
            end = event["end"]
            labels[start - 1 : end + 1] = 1
        self.labels = labels

    def evaluate_sequential(self):
        losses = np.load(config.losses_path)
        predictions = (losses >= config.threshold).astype(np.uint8)
        labels = self.labels
        print(losses)
        print(labels)
        print(predictions)
        results = {}
        kappa = metrics.cohen_kappa_score(labels, predictions)
        results["kappa"] = kappa
        tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions).ravel()
        results["true negative"] = tn
        results["false positive"] = fp
        results["false negative"] = fn
        results["true positive"] = tp
        precision = (tp) / (tp + fp)
        recall = (tp) / (tp + fn)
        f1_score = (2 * recall * precision) / (recall + precision)
        results["precision"] = precision
        results["recall"] = recall
        results["f1 score"] = f1_score
        results["auc"] = metrics.roc_auc_score(labels, predictions)
        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, losses)
        results["auprc"] = metrics.auc(recalls, precisions)
        self.results = results

    def generate_evaluation_plot(self):
        plt.title("Evaluation: " + config.dataset)
        normal_losses = np.load(config.normal_losses_save_path)
        anomalous_losses = np.load(config.anomalous_losses_save_path)
        losses = np.append(normal_losses, anomalous_losses, axis=-1)
        plt.plot(losses)
        plt.ylim((0, config.plot_vertical_limit))
        ymin, ymax = plt.gca().get_ylim()
        threshold_length = normal_losses.shape[-1] + anomalous_losses.shape[-1]
        separation_point = normal_losses.shape[-1]
        plt.hlines(config.threshold, 0, threshold_length, color="r")
        plt.vlines(separation_point, 0, ymax, color="r")
        plt.savefig(config.evaluation_plot_location)

    def evaluate(self):
        normal_losses = np.load(config.normal_losses_save_path)
        anomalous_losses = np.load(config.anomalous_losses_save_path)
        losses = np.append(normal_losses, anomalous_losses, axis=-1)
        normal_labels = np.zeros_like(normal_losses)
        anomalous_labels = np.ones_like(anomalous_losses)
        labels = np.append(normal_labels, anomalous_labels, axis=-1)
        # predictions = losses >= config.threshold
        predictions = losses <= config.threshold
        predictions = predictions.astype(np.uint8)
        results = {}
        kappa = metrics.cohen_kappa_score(labels, predictions)
        results["kappa"] = kappa
        tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions).ravel()
        results["true negative"] = tn
        results["false positive"] = fp
        results["false negative"] = fn
        results["true positive"] = tp
        precision = (tp) / (tp + fp)
        recall = (tp) / (tp + fn)
        f1_score = (2 * recall * precision) / (recall + precision)
        results["precision"] = precision
        results["recall"] = recall
        results["f1 score"] = f1_score
        results["auc"] = metrics.roc_auc_score(labels, predictions)
        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, losses)
        # precision_location = np.where(np.isclose(recalls, config.fixed_recall, atol=0.01))[0][0]
        # results["precision_recall_90"] = precisions[precision_location]
        results["auprc"] = metrics.auc(recalls, precisions)
        plt.figure()
        plt.plot(recalls, precisions, marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(config.auprc_curve_location)
        max_loss = np.max(losses)
        min_loss = np.min(losses)
        bins = np.arange(min_loss, max_loss, (max_loss - min_loss) / 100)
        plt.figure()
        plt.hist(
            normal_losses,
            bins=bins,
            cumulative=False,
            density=True,
            color="b",
            alpha=0.5,
        )  # density = normalized hist, put true for hightly imbalanced datasets
        plt.hist(
            anomalous_losses,
            bins=bins,
            cumulative=False,
            density=True,
            color="r",
            alpha=0.5,
        )
        plt.savefig(config.histogram_location)
        self.results = results

    def show_evaluation_results(self):
        print("Evaluation of: ", config.dataset)
        print("True Negative: ", self.results["true negative"])
        print("False Positive: ", self.results["false positive"])
        print("False Negative: ", self.results["false negative"])
        print("True Positive: ", self.results["true positive"])
        print("Kappa: ", self.results["kappa"])
        print("F1 Score: ", self.results["f1 score"])
        print("Precision: ", self.results["precision"])
        print("Recall: ", self.results["recall"])
        print("AUC: ", self.results["auc"])
        print("AUPRC: ", self.results["auprc"])
        # print("Precision at Recall = 0.9: ", self.results["precision_recall_90"])
