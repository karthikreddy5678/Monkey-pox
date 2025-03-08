import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Activation, LeakyReLU
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define file paths for train, validation, and test directories
train_dir = '/content/drive/MyDrive/mpox/Fold1/Fold1/Fold1/Train'
val_dir = '/content/drive/MyDrive/mpox/Fold1/Fold1/Fold1/Val'
test_dir = '/content/drive/MyDrive/mpox/Fold1/Fold1/Fold1/Test'

# Preprocess the data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Set up generators for training and validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Assuming binary classification (e.g., infected vs. non-infected)
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Important for consistency in predictions
)

# Load the base model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(20)(x)
x = Activation(LeakyReLU(alpha=0.005))(x)
output = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Train the model
EPOCHS = 4
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[lr_callback]
)

# Evaluate the model on the test set
predictions = model.predict(test_generator, verbose=1)

# Convert predictions to binary (0 or 1)
predicted_classes = (predictions >= 0.8).astype(int)

# Get true labels for the test set
true_classes = test_generator.classes

# Compute overall accuracy
overall_accuracy = accuracy_score(true_classes, predicted_classes)
print(f"Overall Accuracy: {overall_accuracy:.2f}")

# Compute classification metrics
precision = precision_score(true_classes, predicted_classes, zero_division=1)
recall = recall_score(true_classes, predicted_classes, zero_division=1)
f1 = f1_score(true_classes, predicted_classes, zero_division=1)

# Print classification report
print("\nClassification Report:")
report = classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys())
print(report)

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(cm)

# Level-wise accuracy for each class
level_wise_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, accuracy in enumerate(level_wise_accuracy):
    print(f"Level-wise Accuracy for Class {i}: {accuracy:.2f}")

# Additional Metrics
print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
