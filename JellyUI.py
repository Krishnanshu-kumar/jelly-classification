import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sv_ttk  # For modern theme
from transformers import TFViTModel, ViTFeatureExtractor


class JellyfishClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Jellyfish Classifier")
        self.root.geometry("900x600")

        # Apply modern theme
        sv_ttk.set_theme("dark")

        # Load the trained models
        try:
            self.resnet_model = load_model('resnet_model.h5')
            self.densenet_model = load_model('densenet_model.h5')
            self.cnn_model = load_model('cnn_model.h5')

            # Load the ViT components
            self.vit_classifier = load_model('classifier_model.h5')
            self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            self.vit_base_model = TFViTModel.from_pretrained('google/vit-base-patch16-224')
        except Exception as e:
            messagebox.showerror("Error",
                                 f"Could not load the trained models. Make sure they exist in the current directory.\nError: {str(e)}")
            root.destroy()
            return

        # Species mapping and information database
        self.species_info = {
            0: {
                "name": "Moon Jellyfish (Aurelia aurita)",
                "details": """Moon jellyfish are characterized by their translucent, moon-like bells and are found in coastal waters worldwide. They typically grow to 25-40cm in diameter and feed mainly on plankton, mollusks, and crustaceans. These jellyfish are bioluminescent and can produce a blue-green light, making them visible at night."""
            },
            1: {
                "name": "Barrel Jellyfish (Rhizostoma pulmo)",
                "details": """The barrel jellyfish is the largest jellyfish species found in UK waters, reaching up to 90cm in diameter. They have eight frilly arms containing small stinging tentacles and feed primarily on plankton. Despite their impressive size, they have a very mild sting."""
            },
            2: {
                "name": "Blue Jellyfish (Cyanea lamarckii)",
                "details": """The blue jellyfish has a distinctive light blue or yellow coloring and can grow up to 30cm. They are found in the North Sea and parts of the Atlantic Ocean, typically in cooler waters. Their tentacles contain powerful stinging cells used to catch fish."""
            },
            3: {
                "name": "Compass Jellyfish (Chrysaora hysoscella)",
                "details": """The compass jellyfish is named for the distinctive compass-like marking on its bell. They can grow up to 30cm in diameter and have 24 tentacles arranged in eight groups. These jellyfish are found in coastal waters and can deliver a painful sting."""
            },
            4: {
                "name": "Lion's Mane Jellyfish (Cyanea capillata)",
                "details": """The lion's mane is one of the largest known species of jellyfish, with bells reaching up to 2 meters in diameter. Their tentacles can grow to exceptional lengths, sometimes exceeding 30 meters. They are found in cooler, northern waters and have a powerful sting."""
            },
            5: {
                "name": "Mauve Stinger Jellyfish (Pelagia noctiluca)",
                "details": """The mauve stinger is a small but potent jellyfish species with a distinctive purple-mauve coloring. They are bioluminescent and can produce bright flashes of light when disturbed. Their powerful sting can cause severe pain and skin reactions."""
            }
        }

        self.setup_gui()

    def setup_gui(self):
        # Create main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Create left and right frames
        left_frame = ttk.Frame(main_container)
        right_frame = ttk.Frame(main_container)

        left_frame.grid(row=0, column=0, padx=5, sticky="nsew")
        right_frame.grid(row=0, column=1, padx=5, sticky="nsew")

        # Image frame
        image_frame = ttk.LabelFrame(left_frame, text="Image", padding="5")
        image_frame.grid(row=0, column=0, pady=5, sticky="nsew")

        self.image_label = ttk.Label(image_frame)
        self.image_label.grid(row=0, column=0, padx=5, pady=5)

        # Control frame
        control_frame = ttk.LabelFrame(left_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=0, pady=5, sticky="ew")

        # Model selection
        self.model_var = tk.StringVar(value="resnet")
        model_frame = ttk.Frame(control_frame)
        model_frame.grid(row=0, column=0, pady=5)

        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(model_frame, text="ResNet", variable=self.model_var, value="resnet").grid(row=0, column=1,
                                                                                                  padx=5)
        ttk.Radiobutton(model_frame, text="DenseNet", variable=self.model_var, value="densenet").grid(row=0, column=2,
                                                                                                      padx=5)
        ttk.Radiobutton(model_frame, text="CNN", variable=self.model_var, value="cnn").grid(row=0, column=3,
                                                                                            padx=5)
        # Added Vision Transformer option
        ttk.Radiobutton(model_frame, text="ViT", variable=self.model_var, value="vit").grid(row=0, column=4,
                                                                                            padx=5)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, pady=5)

        ttk.Button(button_frame, text="Upload Image", command=self.upload_image).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Classify", command=self.classify_image).grid(row=0, column=1, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(right_frame, text="Classification Results", padding="5")
        results_frame.grid(row=0, column=0, pady=5, sticky="nsew")

        self.species_label = ttk.Label(results_frame, text="Species: ")
        self.species_label.grid(row=0, column=0, pady=5, padx=5, sticky="w")

        self.confidence_label = ttk.Label(results_frame, text="Confidence: ")
        self.confidence_label.grid(row=1, column=0, pady=5, padx=5, sticky="w")

        # Details frame
        details_frame = ttk.LabelFrame(right_frame, text="Scientific Details", padding="5")
        details_frame.grid(row=1, column=0, pady=5, sticky="nsew")

        self.details_text = tk.Text(details_frame, wrap=tk.WORD, height=10, width=40)
        self.details_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.details_text.configure(yscrollcommand=scrollbar.set)

        # Configure weights
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)
        details_frame.grid_columnconfigure(0, weight=1)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            # Display image
            image = Image.open(file_path)
            image.thumbnail((300, 300))  # Resize while maintaining aspect ratio
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

    def preprocess_image(self, image_path, for_vit=False):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        if for_vit:
            # For ViT: keep grayscale as a single channel for now
            return img
        else:
            # For CNN, ResNet, DenseNet: normalize and add channels
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)
            return np.expand_dims(img, axis=0)

    def extract_vit_features(self, image):
        # Convert grayscale to 3 channels (RGB)
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)

        # Prepare for feature extractor (normalize and convert to batch)
        image = image.astype('float32') / 255.0

        # Get features using ViT
        encodings = self.feature_extractor([image], return_tensors='tf')
        pixel_values = encodings['pixel_values']

        # Extract features from base model
        features = self.vit_base_model(pixel_values, training=False).last_hidden_state[:, 0, :]

        return features.numpy()

    def get_scientific_details(self, species_id):
        return self.species_info[species_id]["details"]

    def classify_image(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("Error", "Please upload an image first.")
            return

        try:
            # Get selected model
            model_choice = self.model_var.get()

            # Handle classification based on model type
            if model_choice == "vit":
                # Process for ViT model
                img = self.preprocess_image(self.current_image_path, for_vit=True)
                features = self.extract_vit_features(img)
                predictions = self.vit_classifier.predict(features)[0]
                model_name = "Vision Transformer"
            else:
                # Process for CNN/ResNet/DenseNet models
                processed_image = self.preprocess_image(self.current_image_path)

                if model_choice == "resnet":
                    predictions = self.resnet_model.predict(processed_image)[0]
                    model_name = "ResNet"
                elif model_choice == "densenet":
                    predictions = self.densenet_model.predict(processed_image)[0]
                    model_name = "DenseNet"
                else:  # cnn
                    predictions = self.cnn_model.predict(processed_image)[0]
                    model_name = "CNN"

            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100

            # Update results
            species_name = self.species_info[predicted_class]["name"]
            self.species_label.configure(text=f"Species: {species_name}")
            self.confidence_label.configure(text=f"Confidence: {confidence:.2f}% ({model_name})")

            # Get and display scientific details
            self.details_text.delete(1.0, tk.END)
            details = self.get_scientific_details(predicted_class)
            self.details_text.insert(tk.END, details)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during classification: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    root = tk.Tk()
    app = JellyfishClassifierGUI(root)
    root.mainloop()