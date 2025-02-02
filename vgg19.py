import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.io import loadmat

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Download & extract dataset
!wget -q https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -O flowers.tgz
!wget -q https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat -O labels.mat
!mkdir -p flowers && tar -xzf flowers.tgz -C flowers

# Load labels
labels = loadmat("labels.mat")['labels'][0] - 1  # Convert labels to 0-based index
image_dir = "flowers/jpg"
image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])

# Split dataset (50% Train, 25% Validation, 25% Test) - **Random Split Twice**
X_train, X_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.5, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Preprocessing Function
def preprocess_images(image_paths, labels, target_size=(224, 224), batch_size=32, augment=True):
    labels = [str(label) for label in labels]  # Convert labels to strings
    df = pd.DataFrame({"filename": image_paths, "class": labels})

    # Only apply augmentations to training set
    if augment:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(rescale=1.0 / 255)

    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        y_col="class",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

# Prepare Datasets
train_gen = preprocess_images(X_train, y_train, augment=True)
val_gen = preprocess_images(X_val, y_val, augment=False)
test_gen = preprocess_images(X_test, y_test, augment=False)

# Load Pretrained VGG19 (Freeze Early Layers)
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:  # Freeze all layers except last 4
    layer.trainable = False

# Build Improved Model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output_layer = Dense(102, activation='softmax')(x)  # 102 flower classes

model = Model(inputs=base_model.input, outputs=output_layer)

# Compile Model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Learning Rate Adjustment & Early Stopping
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Custom Callback to Track Test Accuracy/Loss per Epoch
class TestMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_gen):
        self.test_gen = test_gen
        self.test_acc = []
        self.test_loss = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_gen, verbose=0)
        self.test_loss.append(test_loss)
        self.test_acc.append(test_acc)
        print(f"\nEpoch {epoch+1}: Test Accuracy = {test_acc:.4f}, Test Loss = {test_loss:.4f}")

# Initialize Callback
test_metrics_callback = TestMetricsCallback(test_gen)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[callbacks, test_metrics_callback]  # Track Test Metrics
)

# Extract Metrics
epochs = range(1, len(history.history['accuracy']) + 1)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
test_acc = test_metrics_callback.test_acc  # Collected per epoch

train_loss = history.history['loss']
val_loss = history.history['val_loss']
test_loss = test_metrics_callback.test_loss  # Collected per epoch

# Plot Accuracy Graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label="Train Accuracy", color='blue')
plt.plot(epochs, val_acc, label="Validation Accuracy", color='orange')
plt.plot(epochs, test_acc, label="Test Accuracy", color='green')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train vs Validation vs Test Accuracy")
plt.legend()

# Plot Cross-Entropy Loss Graph
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label="Train Loss", color='blue')
plt.plot(epochs, val_loss, label="Validation Loss", color='orange')
plt.plot(epochs, test_loss, label="Test Loss", color='green')
plt.xlabel("Epochs")
plt.ylabel("Loss (Cross-Entropy)")
plt.title("Train vs Validation vs Test Loss")
plt.legend()

plt.show()

# Print Final Test Accuracy
print(f"✅ Final Test Accuracy: {test_acc[-1] * 100:.2f}%")

# Save the trained model
model.save("vgg19_flower_classifier_improved.h5")
