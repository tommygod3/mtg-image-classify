import os, sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50

color_classnames = ["Black", "Blue", "Colorless", "Green", "Red",
                    "White"]

type_classnames = ["Artifact",
                    "Creature",
                    "Enchantment",
                    "InstantSorcery",
                    "Land",
                    "Planeswalker"]

combined_classnames = ['Artifact', 'Black', 'Blue', 'Colorless', 'Creature', 'Enchantment', 'Green', 'InstantSorcery', 'Land', 'Planeswalker', 'Red', 'White']

def get_absolute_path(filename):
    return f"{os.path.dirname(os.path.realpath(sys.argv[0]))}/{filename}"

class ArtCNN:
    def __init__(self, dataset_dir, csv_filename, model_filename, class_names, batch_size = 32, num_epochs = 50, img_height = 150, img_width = 150, color_bands = 3):
        self.dataset_dir = dataset_dir
        self.csv_filename = csv_filename
        self.model_filename = model_filename
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.img_height = img_height
        self.img_width = img_width
        self.color_bands = color_bands
        self.class_names = class_names

        self.input_shape = (img_height, img_width, color_bands)

        self.setup_data_generators()
        self.create_model()
        self.train_model()

    def setup_data_generators(self):
        self.dataframe = pd.read_csv(self.csv_filename)

        training_data_limit = int(len(self.dataframe)*0.8)

        self.datagen = ImageDataGenerator(rescale=1./255,
                                          rotation_range=25, width_shift_range=0.1,
                                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                          horizontal_flip=True, fill_mode="nearest")

        self.test_datagen=ImageDataGenerator(rescale=1./255.)

        self.train_generator = self.datagen.flow_from_dataframe(
            dataframe=self.dataframe[:training_data_limit],
            directory=self.dataset_dir,
            x_col="Filename",
            y_col=self.class_names,
            target_size=(self.img_height, self.img_width),
            shuffle=True,
            batch_size=self.batch_size,
            class_mode="other")

        self.validation_generator = self.test_datagen.flow_from_dataframe(
            dataframe=self.dataframe[training_data_limit:],
            directory=self.dataset_dir,
            x_col="Filename",
            y_col=self.class_names,
            target_size=(self.img_height, self.img_width),
            shuffle=True,
            batch_size=self.batch_size,
            class_mode="other")


    def create_model(self):
        # Create CNN
        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation="relu", padding="same", input_shape=self.input_shape),
            BatchNormalization(axis=-1),
            MaxPooling2D(pool_size=(3, 3)),
            Dropout(rate=0.25),

            Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu", padding="same"),
            BatchNormalization(axis=-1),
            Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu", padding="same"),
            BatchNormalization(axis=-1),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(rate=0.25),

            Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation="relu", padding="same"),
            BatchNormalization(axis=-1),
            Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation="relu", padding="same"),
            BatchNormalization(axis=-1),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(rate=0.25),

            Flatten(),
            Dense(units=1024, activation="relu"),
            BatchNormalization(),
            Dropout(rate=0.5),
            Dense(units=len(self.class_names), activation="sigmoid")
        ])

        # Display model summary
        self.model.summary()

        # Compile model
        self.model.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])
    
    def train_model(self):
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size,
            validation_data = self.validation_generator, 
            validation_steps = self.validation_generator.samples // self.validation_generator.batch_size,
            epochs = self.num_epochs)

        self.model.save(self.model_filename)


dataset_dir = get_absolute_path("../download/art/images")
csv_filename = get_absolute_path("../download/art/images.csv")
model_filename = get_absolute_path("model.h5")

print("Combined:")
ArtCNN(dataset_dir=dataset_dir, csv_filename=csv_filename, model_filename=model_filename, class_names=combined_classnames)
