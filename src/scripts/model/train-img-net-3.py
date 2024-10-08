# Multi-Class Classification with MobileNetV2 for Flower Classification

# 1. Import Necessary Packages
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import (
    RandomFlip, RandomRotation, RandomZoom, RandomTranslation,
    RandomContrast, RandomBrightness, BatchNormalization, GlobalMaxPooling2D, Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
)
from tensorflow.keras import mixed_precision
from sklearn.utils import class_weight

# 2. Define Constants and Parameters
BATCH_SIZE = 32  # Reduced batch size for better generalization
IMG_SIZE = (224, 224)  # Increased image size for better feature representation
IMG_SHAPE = IMG_SIZE + (3,)
DIRECTORY = "../flowers"  # Replace with your dataset directory path
BASE_LEARNING_RATE = 0.001  # Base learning rate
INITIAL_EPOCHS = 9  # Increased epochs 9
FINE_TUNE_EPOCHS = 10  # 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
FINE_TUNE_AT = 100  # Adjusted for finer control during fine-tuning
NUM_CLASSES = 15     # Number of classes

# 3. Enable Mixed Precision
mixed_precision.set_global_policy('mixed_float16')

# 4. Create Training and Validation Datasets
train_dataset = image_dataset_from_directory(
    DIRECTORY,
    validation_split=0.2,
    subset='training',
    seed=42,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True  # Ensure shuffling
)

validation_dataset = image_dataset_from_directory(
    DIRECTORY,
    validation_split=0.2,
    subset='validation',
    seed=42,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 5. Optimize Data Pipeline with Caching and Prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# 6. Define Enhanced Data Augmentation
def data_augmenter():
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomFlip("vertical"),          
        RandomRotation(0.2),              
        RandomZoom(0.2),
        RandomTranslation(0.2, 0.2),
        RandomContrast(0.2),              
    ])
    return data_augmentation

augmenter = data_augmenter()

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze the base model initially

# 8. Build the Model
def multiclass_flower_model(image_shape=IMG_SIZE, data_augmentation=augmenter, num_classes=NUM_CLASSES):
    inputs = tf.keras.Input(shape=image_shape + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    
    # Apply both pooling layers independently to the same tensor for more diverse feature extraction
    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
    
    # Concatenate the outputs of both pooling layers
    x = Concatenate()([gap, gmp])
    
    x = BatchNormalization()(x)  # Added Batch Normalization
    x = tfl.Dropout(0.5)(x)  # Increased dropout rate
    x = tfl.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Increased Dense units
    x = BatchNormalization()(x)  # Added Batch Normalization
    x = tfl.Dropout(0.5)(x)  # Additional dropout
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)  # Set dtype to float32 for numerical stability
    model = tf.keras.Model(inputs, outputs)
    return model

# 9. Define Distributed Strategy (Optional)
strategy = tf.distribute.get_strategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = multiclass_flower_model(IMG_SIZE, augmenter, NUM_CLASSES)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE * 0.1),
        metrics=['accuracy']
    )

model.summary()

# 10. Define Callbacks for Early Stopping, Learning Rate Reduction, TensorBoard, and Warm-Up
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Increased patience
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,  # Increased patience
    min_lr=1e-6,
    verbose=1
)

tensorboard_callback = TensorBoard(log_dir='logs/', histogram_freq=1)

def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr * 1.1  # Warm-up for first 5 epochs
    else:
        return lr

warmup_lr = LearningRateScheduler(lr_scheduler)

callbacks = [early_stopping, reduce_lr, tensorboard_callback, warmup_lr]

# 11. Compute Class Weights
# Flatten the dataset labels
train_labels = np.concatenate([y for x, y in train_dataset], axis=0)
class_weights_values = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights_values))
print("Class Weights: ", class_weights)

# 12. Train the Model with Callbacks and Class Weights
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights  # Added class weights
)

# 13. Plot Training and Validation Metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 10))

# Accuracy Plot
plt.subplot(2, 1, 1)
plt.plot(range(len(acc)), acc, label='Training Accuracy')
plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Training and Validation Accuracy')

# Loss Plot
plt.subplot(2, 1, 2)
plt.plot(range(len(loss)), loss, label='Training Loss')
plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.ylim([0, max(max(loss), max(val_loss))])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.show()

# 14. Fine-Tuning the Model
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Freeze the first FINE_TUNE_AT layers
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Re-compile the model with a lower learning rate for fine-tuning
with strategy.scope():
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE * 0.01),
        metrics=['accuracy']
    )

# Continue Training (Fine-Tuning) with Callbacks
history_fine = model.fit(
    train_dataset,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
    callbacks=callbacks,
    class_weight=class_weights
)

# Update the accuracy and loss values
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

# 15. Plot Training and Validation Metrics After Fine-Tuning
plt.figure(figsize=(12, 10))

# Accuracy Plot
plt.subplot(2, 1, 1)
plt.plot(range(len(acc)), acc, label='Training Accuracy')
plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
plt.axvline(x=INITIAL_EPOCHS-1, color='r', linestyle='--', label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Training and Validation Accuracy')

# Loss Plot
plt.subplot(2, 1, 2)
plt.plot(range(len(loss)), loss, label='Training Loss')
plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
plt.axvline(x=INITIAL_EPOCHS-1, color='r', linestyle='--', label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.ylim([0, max(max(loss), max(val_loss))])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.show()

# 16. Save the Model (Optional)
model.save('flower_classifier_model_enhanced.h5')

# 17. Evaluate the Model on Validation Data
loss, accuracy = model.evaluate(validation_dataset)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# 18. Make Predictions (Optional)
# Define class names (ensure this matches your dataset)
class_names = train_dataset.class_names

# Example of making predictions on a batch
image_batch, label_batch = next(iter(validation_dataset))
predictions = model.predict(image_batch)  # Shape: (batch_size, NUM_CLASSES)
predicted_classes = np.argmax(predictions, axis=1)

for i in range(len(predicted_classes)):
    true_label = class_names[label_batch[i]]
    predicted_label = class_names[predicted_classes[i]]
    probability = predictions[i][predicted_classes[i]]
    print(f"Image {i+1}: True Label: {true_label} | Predicted: {predicted_label} (Probability: {probability:.4f})")