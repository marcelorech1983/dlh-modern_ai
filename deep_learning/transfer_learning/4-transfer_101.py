#!/usr/bin/env python3
"""Transfer learning on Caltech-101 with a pretrained MobileNetV2."""
import tensorflow as tf
from tensorflow import keras


def train_transfer_model():
    """Train a Caltech-101 classifier and save caltech101_model.h5."""
    keras.utils.set_random_seed(42)

    # Read the images and split them 80% train / 20% validation.
    train_ds = keras.utils.image_dataset_from_directory(
        "101_ObjectCategories", validation_split=0.2, subset="training",
        seed=42, image_size=(224, 224), batch_size=32)
    val_ds = keras.utils.image_dataset_from_directory(
        "101_ObjectCategories", validation_split=0.2, subset="validation",
        seed=42, image_size=(224, 224), batch_size=32)

    num_classes = len(train_ds.class_names)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Augmentation, applied to training images only.
    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=42),
        keras.layers.RandomRotation(0.15, seed=42),
        keras.layers.RandomZoom(0.15, seed=42),
        keras.layers.RandomContrast(0.1, seed=42),
    ])

    # Frozen backbone plus a new classification head.
    base = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = augment(inputs)

    x = keras.layers.Rescaling(1. / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2, seed=42)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2),
    ]

    # Train the head, backbone frozen.
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=15,
              callbacks=callbacks)

    # Unfreeze the last 20 layers and fine-tune slowly.
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    for layer in base.layers[-20:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    # Recompiling.
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              callbacks=callbacks)

    # Report and save.
    loss, accuracy = model.evaluate(val_ds)
    print("Validation accuracy: {:.2%}".format(accuracy))
    model.save("caltech101_model.h5")
    return model


if __name__ == "__main__":
    train_transfer_model()
