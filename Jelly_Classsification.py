import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateaua

# Check if models and histories already exist
models_trained = os.path.exists('resnet_model.h5') and os.path.exists('densenet_model.h5') and \
                 os.path.exists('cnn_model.h5') and os.path.exists('model_histories.pkl')

if not models_trained:
    # Define the path to the dataset folders
    Moon_jellyfish_folder = "dataset/Moon_jellyfish"
    barrel_jellyfish_folder = "dataset/barrel_jellyfish"
    blue_jellyfish_folder = "dataset/blue_jellyfish"
    compass_jellyfish_folder = "dataset/compass_jellyfish"
    lions_mane_jellyfish_folder = "dataset/lions_mane_jellyfish"
    mauve_stinger_jellyfish_folder = "dataset/mauve_stinger_jellyfish"

    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (224, 224))
                images.append(img)
        return images

    # Load images for each species
    print("Loading images...")
    Moon_images = load_images_from_folder(Moon_jellyfish_folder)
    barrel_images = load_images_from_folder(barrel_jellyfish_folder)
    blue_images = load_images_from_folder(blue_jellyfish_folder)
    compass_images = load_images_from_folder(compass_jellyfish_folder)
    lions_mane_images = load_images_from_folder(lions_mane_jellyfish_folder)
    mauve_stinger_images = load_images_from_folder(mauve_stinger_jellyfish_folder)

    # Create labels
    Moon_labels = [0] * len(Moon_images)
    barrel_labels = [1] * len(barrel_images)
    blue_labels = [2] * len(blue_images)
    compass_labels = [3] * len(compass_images)
    lions_mane_labels = [4] * len(lions_mane_images)
    mauve_stinger_labels = [5] * len(mauve_stinger_images)

    # Combine all labels
    all_labels = Moon_labels + barrel_labels + blue_labels + compass_labels + lions_mane_labels + mauve_stinger_labels

    # Print dataset statistics
    print('Moon jellyfish images:', len(Moon_images))
    print('Barrel jellyfish images:', len(barrel_images))
    print('Blue jellyfish images:', len(blue_images))
    print('Compass jellyfish images:', len(compass_images))
    print('Lion\'s mane jellyfish images:', len(lions_mane_images))
    print('Mauve stinger jellyfish images:', len(mauve_stinger_images))

    # Plot data distribution
    plt.figure(figsize=(8, 3))
    sns.countplot(x=all_labels, hue=all_labels)
    plt.title('Data Distribution by Species')
    plt.xlabel('Jellyfish species')
    plt.ylabel('Count')
    plt.show()

    # Prepare data
    X = np.array(Moon_images + barrel_images + blue_images + compass_images + lions_mane_images + mauve_stinger_images)
    y = np.array(all_labels)

    # Normalize and encode
    X = X.astype('float32') / 255.0
    y = to_categorical(y, 6)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save test data for later evaluation
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # Define input shapes
    input_shape = (224, 224, 3)

    def resize_images(images, input_shape):
        resized_images = []
        for img in images:
            img_resized = cv2.resize(img, (input_shape[0], input_shape[1]))
            img_resized = np.expand_dims(img_resized, axis=-1)
            img_resized = np.repeat(img_resized, 3, axis=-1)
            resized_images.append(img_resized)
        return np.array(resized_images)

    # Resize images for all models
    X_train_resized = resize_images(X_train, input_shape)
    X_test_resized = resize_images(X_test, input_shape)

    # Create and compile ResNet model
    print("Creating ResNet model...")
    resnet_base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    resnet_base_model.trainable = False
    resnet_global_avg_pooling = GlobalAveragePooling2D()(resnet_base_model.output)
    resnet_output = Dense(6, activation='softmax')(resnet_global_avg_pooling)
    resnet_model = Model(inputs=resnet_base_model.input, outputs=resnet_output)
    resnet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Create and compile DenseNet model
    print("Creating DenseNet model...")
    densenet_base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    densenet_base_model.trainable = False
    densenet_global_avg_pooling = GlobalAveragePooling2D()(densenet_base_model.output)
    densenet_output = Dense(6, activation='softmax')(densenet_global_avg_pooling)
    densenet_model = Model(inputs=densenet_base_model.input, outputs=densenet_output)
    densenet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Create and compile custom CNN model
    print("Creating CNN model...")
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    cnn_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-7)
    callbacks = [early_stopping, lr_scheduler]

    # Train models
    print("Training ResNet model...")
    resnet_history = resnet_model.fit(
        X_train_resized, y_train,
        batch_size=64,
        epochs=30,
        validation_split=0.2,
        callbacks=callbacks
    )

    print("\nTraining DenseNet model...")
    densenet_history = densenet_model.fit(
        X_train_resized, y_train,
        batch_size=64,
        epochs=30,
        validation_split=0.2,
        callbacks=callbacks
    )

    print("\nTraining CNN model...")
    cnn_history = cnn_model.fit(
        X_train_resized, y_train,
        batch_size=64,
        epochs=30,
        validation_split=0.2,
        callbacks=callbacks
    )

    # Save models and histories
    print("Saving models and training histories...")
    resnet_model.save('resnet_model.h5')
    densenet_model.save('densenet_model.h5')
    cnn_model.save('cnn_model.h5')

    # Save histories
    histories = {
        'resnet_history': resnet_history.history,
        'densenet_history': densenet_history.history,
        'cnn_history': cnn_history.history
    }
    with open('model_histories.pkl', 'wb') as f:
        pickle.dump(histories, f)

else:
    # Load existing models and histories
    print("Loading pre-trained models and histories...")
    resnet_model = load_model('resnet_model.h5')
    densenet_model = load_model('densenet_model.h5')
    cnn_model = load_model('cnn_model.h5')

    with open('model_histories.pkl', 'rb') as f:
        histories = pickle.load(f)

    # Load test data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    # Resize test data
    input_shape = (224, 224, 3)

    def resize_images(images, input_shape):
        resized_images = []
        for img in images:
            img_resized = cv2.resize(img, (input_shape[0], input_shape[1]))
            img_resized = np.expand_dims(img_resized, axis=-1)
            img_resized = np.repeat(img_resized, 3, axis=-1)
            resized_images.append(img_resized)
        return np.array(resized_images)

    X_test_resized = resize_images(X_test, input_shape)

# Evaluate models
print("\nEvaluating models...")
resnet_loss, resnet_accuracy = resnet_model.evaluate(X_test_resized, y_test)
densenet_loss, densenet_accuracy = densenet_model.evaluate(X_test_resized, y_test)
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_resized, y_test)

print("\nModel Evaluation Results:")
print(f"ResNet Test Accuracy: {resnet_accuracy:.4f}")
print(f"DenseNet Test Accuracy: {densenet_accuracy:.4f}")
print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")

# Plot training history
def plot_training_history(histories):
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Model Training Histories', fontsize=16)

    # Plot ResNet histories
    axes[0, 0].plot(histories['resnet_history']['loss'], label='Training Loss')
    axes[0, 0].plot(histories['resnet_history']['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('ResNet Learning Curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(histories['resnet_history']['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(histories['resnet_history']['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('ResNet Accuracy Curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Plot DenseNet histories
    axes[1, 0].plot(histories['densenet_history']['loss'], label='Training Loss')
    axes[1, 0].plot(histories['densenet_history']['val_loss'], label='Validation Loss')
    axes[1, 0].set_title('DenseNet Learning Curve')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()

    axes[1, 1].plot(histories['densenet_history']['accuracy'], label='Training Accuracy')
    axes[1, 1].plot(histories['densenet_history']['val_accuracy'], label='Validation Accuracy')
    axes[1, 1].set_title('DenseNet Accuracy Curve')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()

    # Plot CNN histories
    axes[2, 0].plot(histories['cnn_history']['loss'], label='Training Loss')
    axes[2, 0].plot(histories['cnn_history']['val_loss'], label='Validation Loss')
    axes[2, 0].set_title('CNN Learning Curve')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Loss')
    axes[2, 0].legend()

    axes[2, 1].plot(histories['cnn_history']['accuracy'], label='Training Accuracy')
    axes[2, 1].plot(histories['cnn_history']['val_accuracy'], label='Validation Accuracy')
    axes[2, 1].set_title('CNN Accuracy Curve')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Accuracy')
    axes[2, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('model_training_histories.png')
    plt.show()

# Plot training histories
print("\nPlotting training histories...")
plot_training_history(histories)
