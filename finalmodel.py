import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Input, SeparableConv2D, BatchNormalization, ReLU, 
                                     MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Multiply, Reshape)
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. GPU Memory Management
# ----------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs detected. Running on CPU.")

# ----------------------------------------------------------
# 2. Paths and Hyperparameters
# ----------------------------------------------------------
DATASET_PATH = "DIAT-uSAT_dataset"  # Update this to the path of your dataset
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
EPOCHS = 100  # Increased epochs with proper early stopping
INITIAL_LR = 3e-4  # Increased initial learning rate

# ----------------------------------------------------------
# 3. Data Augmentation with Balanced Sampling
# ----------------------------------------------------------
# More aggressive data augmentation for better generalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,        # Increased rotation
    width_shift_range=0.3,    # Increased shift
    height_shift_range=0.3,   # Increased shift
    shear_range=0.2,          # Increased shear
    zoom_range=0.3,           # Increased zoom
    horizontal_flip=True,
    vertical_flip=True,       # Added vertical flip
    brightness_range=[0.7, 1.3], # Added brightness variation
    fill_mode='nearest'
)

# Load data and get class distribution
train_generator_raw = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = validation_generator.num_classes
class_indices = validation_generator.class_indices
class_list = list(class_indices.keys())
print(f"Found {num_classes} classes with mapping: {class_indices}")

# Compute class weights to handle class imbalance
train_labels = train_generator_raw.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# Define mixup function with balanced alpha for more stable training
def mixup(batch_x, batch_y, alpha=0.4):  # Increased alpha for more mixing
    lam = np.random.beta(alpha, alpha, size=batch_x.shape[0])
    lam = np.maximum(lam, 1-lam)  # Ensure lambda is always at least 0.5
    lam = lam.reshape(-1, 1, 1, 1)
    
    batch_size = batch_x.shape[0]
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
    lam = lam.reshape(-1, 1)
    mixed_y = lam * batch_y + (1 - lam) * batch_y[index]
    return mixed_x, mixed_y

def generator_with_mixup(generator, alpha=0.4):
    while True:
        batch_x, batch_y = next(generator)
        yield mixup(batch_x, batch_y, alpha)

train_gen = generator_with_mixup(train_generator_raw, alpha=0.4)

# ----------------------------------------------------------
# 4. Custom Loss Function (Focal Loss with Label Smoothing)
# ----------------------------------------------------------
def focal_loss_with_smoothing(gamma=2.0, alpha=0.25, label_smoothing=0.1):
    """
    Focal Loss with label smoothing to prevent overconfidence.
    """
    def loss_fn(y_true, y_pred):
        # Apply label smoothing
        y_true = y_true * (1 - label_smoothing) + label_smoothing / num_classes
        
        # Focal loss implementation
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1. - y_pred, gamma) * y_true
        focal_loss = weight * cross_entropy
        return K.sum(focal_loss, axis=-1)
    return loss_fn

loss_fn = focal_loss_with_smoothing(gamma=2.0, alpha=0.25, label_smoothing=0.1)

# ----------------------------------------------------------
# 5. SE Block with Improved Reduction Ratio
# ----------------------------------------------------------
def se_block(input_tensor, reduction=8):  # Changed from 16 to 8 for better feature recalibration
    """
    Enhanced Squeeze-and-Excitation block with improved reduction ratio
    """
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(max(filters // reduction, 4), activation='relu', use_bias=False)(se)  # Ensure minimum width
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    return Multiply()([input_tensor, se])

# ----------------------------------------------------------
# 6. Improved Lightweight CNN Model Architecture
# ----------------------------------------------------------
def build_improved_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=num_classes):
    """
    Enhanced model with residual connections and deeper architecture
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution with larger filter
    x = SeparableConv2D(64, (5, 5), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    x_skip1 = x  # Store for skip connection
    x = MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    x_skip2 = x  # Store for skip connection
    x = MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 4 with skip connection from Block 2
    skip2_down = MaxPooling2D((2, 2))(x_skip2)
    skip2_down = MaxPooling2D((2, 2))(skip2_down)
    x = tf.keras.layers.Concatenate()([x, skip2_down])
    
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    
    # Global feature aggregation with attention
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    # Two-stage classification head
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# Build the improved model
model = build_improved_model()

# Use Adam with weight decay instead of AdamW (for older TF versions)
# We'll implement weight decay manually in the optimizer
optimizer = Adam(
    learning_rate=INITIAL_LR,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7,
    clipnorm=1.0  # Gradient clipping
)

# Create a weight decay callback to manually apply weight decay after each batch
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, weight_decay=1e-4):
        self.weight_decay = weight_decay
        
    def on_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                # Apply weight decay to kernel weights only (not biases)
                weights[0] = weights[0] * (1 - self.weight_decay)
                layer.set_weights(weights)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

model.summary()

# ----------------------------------------------------------
# 7. Improved Learning Rate Scheduling
# ----------------------------------------------------------
# Combine cosine annealing with reduce on plateau for better learning rate adjustment
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

def cosine_annealing_warm_restarts(epoch, initial_lr=INITIAL_LR, cycle_length=10, min_lr=1e-6):
    """Cosine annealing with warm restarts"""
    epoch = epoch % cycle_length
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / cycle_length))

lr_scheduler = LearningRateScheduler(
    lambda epoch: cosine_annealing_warm_restarts(epoch),
    verbose=1
)

# ----------------------------------------------------------
# 8. Training Callbacks with Improved Early Stopping
# ----------------------------------------------------------
checkpoint_filepath = 'model_outputs/best_model.h5'
os.makedirs('model_outputs', exist_ok=True)

# Add the weight decay callback
weight_decay_callback = WeightDecayCallback(weight_decay=1e-4)

callbacks = [
    # More patience for early stopping to allow the model to learn
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    reduce_lr,
    lr_scheduler,
    weight_decay_callback
]

# ----------------------------------------------------------
# 9. Training with Class Weights
# ----------------------------------------------------------
print("Starting model training with class weights...")
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_generator_raw),
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Save final model
model.save('model_outputs/final_model.h5')

# ----------------------------------------------------------
# 10. Evaluation with Improved Metrics
# ----------------------------------------------------------
print("Evaluating model on validation set...")
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_accuracy:.4f}")

# Generate predictions with temperature scaling for better calibration
validation_generator.reset()
logits = model.predict(validation_generator, verbose=1)

# Apply temperature scaling (T=1.5) to soften predictions
temperature = 1.5
predictions = np.exp(np.log(logits) / temperature)
predictions = predictions / np.sum(predictions, axis=1, keepdims=True)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

print("Detailed Classification Report (10 Classes):")
print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))

# ----------------------------------------------------------
# 11. Improved Aggregation into Broad Groups
# ----------------------------------------------------------
# Define a mapping for the 10 classes to main groups with confidence weighting
GROUP_MAPPING = {
    '2_blade_rotor_1': 'drone',
    '2_blade_rotor_2': 'drone',
    '3_long_blades_rotor': 'drone',
    '3_short_blade_rotor_1': 'drone',
    '3_short_blade_rotor_2': 'drone',
    'Bird': 'bird',
    'Bird+2_Blade_rotor_1': 'ambiguous',
    'Bird+2_Blade_rotor_2': 'ambiguous',
    'drone_1': 'drone',
    'drone_2': 'drone'
}

# Group confidence weighting - how strongly each class contributes to its group
GROUP_CONFIDENCE = {
    '2_blade_rotor_1': 1.0,
    '2_blade_rotor_2': 1.0,
    '3_long_blades_rotor': 1.0,
    '3_short_blade_rotor_1': 1.0,
    '3_short_blade_rotor_2': 1.0,
    'Bird': 1.0,
    'Bird+2_Blade_rotor_1': {'bird': 0.6, 'drone': 0.4},  # Bird is more dominant
    'Bird+2_Blade_rotor_2': {'bird': 0.6, 'drone': 0.4},  # Bird is more dominant
    'drone_1': 1.0,
    'drone_2': 1.0
}

def improved_aggregate_predictions(pred_probs, class_labels, group_mapping, group_confidence):
    """
    Aggregate predicted probabilities with weighted confidence for each group.
    """
    group_scores = {'bird': 0.0, 'drone': 0.0}
    
    for i, label in enumerate(class_labels):
        prob = pred_probs[i]
        group = group_mapping.get(label, 'unknown')
        
        if group == 'bird' or group == 'drone':
            group_scores[group] += prob * 1.0  # Full confidence
        elif group == 'ambiguous':
            # Split according to confidence weights
            conf = group_confidence.get(label, {'bird': 0.5, 'drone': 0.5})
            group_scores['bird'] += prob * conf['bird']
            group_scores['drone'] += prob * conf['drone']
    
    return group_scores['bird'], group_scores['drone']

# Process predictions for binary classification
binary_true = []
binary_pred = []
binary_confidence = []

for i in range(len(true_classes)):
    true_label = class_labels[true_classes[i]]
    true_group = GROUP_MAPPING.get(true_label, 'unknown')
    
    # If ground truth is ambiguous, use the dominant group in the confidence mapping
    if true_group == 'ambiguous':
        conf = GROUP_CONFIDENCE.get(true_label, {'bird': 0.5, 'drone': 0.5})
        true_group = 'bird' if conf['bird'] > conf['drone'] else 'drone'
    
    pred_vector = predictions[i]
    bird_score, drone_score = improved_aggregate_predictions(
        pred_vector, class_labels, GROUP_MAPPING, GROUP_CONFIDENCE
    )
    
    binary_pred.append('bird' if bird_score > drone_score else 'drone')
    binary_true.append(true_group)
    
    # Store confidence level (difference between highest and second highest score)
    confidence = abs(bird_score - drone_score)
    binary_confidence.append(confidence)

from sklearn.metrics import classification_report, confusion_matrix

print("\nBroad Classification Report (Drone vs. Bird):")
print(classification_report(binary_true, binary_pred, target_names=['bird', 'drone'], zero_division=0))

# Print confusion matrix for binary classification
cm = confusion_matrix(
    [1 if x == 'bird' else 0 for x in binary_true],
    [1 if x == 'bird' else 0 for x in binary_pred]
)
print("\nConfusion Matrix (Binary Classification):")
print(" " * 15 + "Predicted")
print(" " * 15 + "Drone  Bird")
print("Actual  Drone  {:5d}  {:5d}".format(cm[0][0], cm[0][1]))
print("        Bird   {:5d}  {:5d}".format(cm[1][0], cm[1][1]))

# ----------------------------------------------------------
# 12. Plot Training History
# ----------------------------------------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot learning rate
plt.subplot(1, 3, 3)
lr_history = [cosine_annealing_warm_restarts(i) for i in range(len(history.history['loss']))]
plt.plot(lr_history)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.savefig('model_outputs/training_history.png')
plt.show()

# ----------------------------------------------------------
# 13. Save Classification Performance Visualization
# ----------------------------------------------------------
# Plot confusion matrix as a heatmap
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(12, 10))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (10 Classes)')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model_outputs/confusion_matrix.png')
plt.show()

# Plot confidence distribution for correct vs. incorrect predictions
plt.figure(figsize=(10, 6))
correct = np.array(predicted_classes) == np.array(true_classes)
correct_conf = np.max(predictions[correct], axis=1) if any(correct) else []
incorrect_conf = np.max(predictions[~correct], axis=1) if any(~correct) else []

if len(correct_conf) > 0:
    plt.hist(correct_conf, bins=20, alpha=0.5, label='Correct Predictions')
if len(incorrect_conf) > 0:
    plt.hist(incorrect_conf, bins=20, alpha=0.5, label='Incorrect Predictions')
    
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.title('Confidence Distribution: Correct vs. Incorrect Predictions')
plt.legend()
plt.savefig('model_outputs/confidence_distribution.png')
plt.show()