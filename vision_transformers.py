import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from transformers import TFViTModel, ViTFeatureExtractor

# Define dataset folders
Moon_jellyfish_folder = "dataset/Moon_jellyfish"
barrel_jellyfish_folder = "dataset/barrel_jellyfish"
blue_jellyfish_folder = "dataset/blue_jellyfish"
compass_jellyfish_folder = "dataset/compass_jellyfish"
lions_mane_jellyfish_folder = "dataset/lions_mane_jellyfish"
mauve_stinger_jellyfish_folder = "dataset/mauve_stinger_jellyfish"


# Function to load images from a folder in grayscale
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
    return images


# Load images for each jellyfish species
print("Loading images...")
Moon_images = load_images_from_folder(Moon_jellyfish_folder)
barrel_images = load_images_from_folder(barrel_jellyfish_folder)
blue_images = load_images_from_folder(blue_jellyfish_folder)
compass_images = load_images_from_folder(compass_jellyfish_folder)
lions_mane_images = load_images_from_folder(lions_mane_jellyfish_folder)
mauve_stinger_images = load_images_from_folder(mauve_stinger_jellyfish_folder)

# Create labels for each species (0 to 5)
Moon_labels = [0] * len(Moon_images)
barrel_labels = [1] * len(barrel_images)
blue_labels = [2] * len(blue_images)
compass_labels = [3] * len(compass_images)
lions_mane_labels = [4] * len(lions_mane_images)
mauve_stinger_labels = [5] * len(mauve_stinger_images)

all_labels = Moon_labels + barrel_labels + blue_labels + compass_labels + lions_mane_labels + mauve_stinger_labels

# Display dataset statistics
print('Moon jellyfish images:', len(Moon_images))
print('Barrel jellyfish images:', len(barrel_images))
print('Blue jellyfish images:', len(blue_images))
print('Compass jellyfish images:', len(compass_images))
print("Lion's mane jellyfish images:", len(lions_mane_images))
print('Mauve stinger jellyfish images:', len(mauve_stinger_images))

# Plot data distribution
plt.figure(figsize=(8, 3))
sns.countplot(x=all_labels, hue=all_labels, palette='viridis', legend=False)
plt.title('Data Distribution by Species')
plt.xlabel('Jellyfish Species')
plt.ylabel('Count')
plt.show()

# Combine images and labels into arrays
X = np.array(Moon_images + barrel_images + blue_images + compass_images + lions_mane_images + mauve_stinger_images)
y = np.array(all_labels)
y = to_categorical(y, num_classes=6)  # One-hot encode labels for 6 classes

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for later use
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)


# Function to prepare grayscale images for ViT (convert to 3 channels)
def prepare_images_for_vit(images):
    images = np.expand_dims(images, axis=-1)  # Add channel dimension: (samples, 224, 224, 1)
    images = np.repeat(images, 3, axis=-1)  # Repeat grayscale to 3 channels: (samples, 224, 224, 3)
    return images


# Prepare training and test images
X_train_resized = prepare_images_for_vit(X_train)
X_test_resized = prepare_images_for_vit(X_test)

# Create lists of images for the feature extractor
X_train_list = [img for img in X_train_resized]
X_test_list = [img for img in X_test_resized]

# Initialize ViT feature extractor and base model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
base_model = TFViTModel.from_pretrained('google/vit-base-patch16-224')


# Function to extract features from images
def extract_features(images):
    # Process images in batches to avoid memory issues
    batch_size = 32
    features_list = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        # Preprocess images
        encodings = feature_extractor(batch_images, return_tensors='tf')
        pixel_values = encodings['pixel_values']

        # Extract features
        batch_features = base_model(pixel_values, training=False).last_hidden_state[:, 0, :]
        features_list.append(batch_features.numpy())

    # Concatenate all batches
    return np.vstack(features_list)


# Extract features if not already done
if not os.path.exists('train_features.npy'):
    print("Extracting features from training images...")
    train_features = extract_features(X_train_list)
    np.save('train_features.npy', train_features)

    print("Extracting features from test images...")
    test_features = extract_features(X_test_list)
    np.save('test_features.npy', test_features)
else:
    print("Loading pre-extracted features...")
    train_features = np.load('train_features.npy')
    test_features = np.load('test_features.npy')


# Create a simple classifier model
def create_classifier_model(input_shape):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(6, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


# Check if model exists
if not os.path.exists('classifier_model.h5'):
    print("Creating and training classifier model...")
    classifier_model = create_classifier_model(train_features.shape[1])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-7)

    # Train the model
    history = classifier_model.fit(
        train_features, y_train,
        batch_size=64,
        epochs=30,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Save the model and history
    classifier_model.save('classifier_model.h5')
    with open('classifier_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
else:
    # Load the model and history
    classifier_model = load_model('classifier_model.h5')
    with open('classifier_history.pkl', 'rb') as f:
        history_dict = pickle.load(f)

# Evaluate the model
print("\nEvaluating classifier model...")
loss, accuracy = classifier_model.evaluate(test_features, y_test)
print(f"Classifier Test Accuracy: {accuracy:.4f}")


# Plot training history
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('ViT Training History', fontsize=16)

    # Plot loss
    axes[0].plot(history['loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Learning Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('classifier_training_history.png')
    plt.show()


# Plot the training history
print("\nPlotting training history...")
with open('classifier_history.pkl', 'rb') as f:
    history_dict = pickle.load(f)
plot_training_history(history_dict)