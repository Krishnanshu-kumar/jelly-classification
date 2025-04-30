import matplotlib.pyplot as plt
import os
import pickle
def plot_training(histories):
    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    models = ['resnet_history', 'densenet_history', 'cnn_history', 'vit_history']
    titles = ['ResNet', 'DenseNet', 'CNN', 'ViT']

    for i, model in enumerate(models):
        axes[i, 0].plot(histories[model]['loss'], label='Training Loss')
        axes[i, 0].plot(histories[model]['val_loss'], label='Validation Loss')
        axes[i, 0].set_title(f'{titles[i]} Learning Curve')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].legend()

        axes[i, 1].plot(histories[model]['accuracy'], label='Training Accuracy')
        axes[i, 1].plot(histories[model]['val_accuracy'], label='Validation Accuracy')
        axes[i, 1].set_title(f'{titles[i]} Accuracy Curve')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].set_ylabel('Accuracy')
        axes[i, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('model_training_histories.png')
    plt.show()
# Load histories from both files if available
history_dict = {}

if os.path.exists('models/model_histories.pkl'):
    with open('models/model_histories.pkl', 'rb') as f:
        history_dict.update(pickle.load(f))  # Load ResNet, DenseNet, and CNN histories

if os.path.exists('classifier_history.pkl'):
    with open('classifier_history.pkl', 'rb') as f:
        history_dict['vit_history'] = pickle.load(f)  # Load ViT history separately


plot_training(history_dict)