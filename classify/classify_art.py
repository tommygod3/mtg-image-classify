import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

color_classnames = ["Black", "Blue", "Colorless", "Green", "Red",
                    "White"]

type_classnames = ["Artifact",
                    "Creature",
                    "Enchantment",
                    "InstantSorcery",
                    "Land",
                    "Planeswalker"]

class ArtCNN:
    def __init__(self, dataset_dir, model_filename, class_names, batch_size = 32, num_epochs = 10, img_height = 150, img_width = 150, color_bands = 3):
        self.dataset_dir = dataset_dir
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
        self.datagen = ImageDataGenerator(rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)

        self.train_generator = self.datagen.flow_from_directory(
            directory=self.dataset_dir,
            target_size=(self.img_height, self.img_width),
            shuffle=True,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training")

        self.validation_generator = self.datagen.flow_from_directory(
            directory=self.dataset_dir,
            target_size=(self.img_height, self.img_width),
            shuffle=True,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation")


    def create_model(self):
        # Create CNN
        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation="relu", input_shape=self.input_shape),
            MaxPooling2D(),
            Dropout(rate=0.25),
            Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"),
            MaxPooling2D(),
            Dropout(rate=0.25),
            Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation="relu"),
            MaxPooling2D(),
            Dropout(rate=0.25),
            Flatten(),
            Dense(units=128, activation="relu"),
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
            steps_per_epoch = self.train_generator.samples // self.batch_size,
            validation_data = self.validation_generator, 
            validation_steps = self.validation_generator.samples // self.batch_size,
            epochs = self.num_epochs)

        self.model.save(self.model_filename)

print("Color: ")
ArtCNN(dataset_dir="../download/art/color", model_filename="color_model.h5", class_names=color_classnames)
print("Type: ")
ArtCNN(dataset_dir="../download/art/type", model_filename="type_model.h5", class_names=type_classnames)
