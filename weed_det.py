import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate synthetic NDVI data (replace this with your actual data loading code)
def generate_synthetic_data(num_samples, image_size):
    data = []
    labels = []
    for _ in range(num_samples):
        ndvi_image = np.random.rand(*image_size)
        weed_intensity = np.random.uniform(0, 1)
        data.append(ndvi_image)
        labels.append(int(weed_intensity > 0.5))  # Binary label based on weed intensity threshold
    return np.array(data), np.array(labels)

image_size = (128, 128)  # Adjust according to your image size
num_samples = 1000
data, labels = generate_synthetic_data(num_samples, image_size)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Flatten images and prepare them for training
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train a Random Forest classifier (you can use other classifiers too)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_flat, y_train)

# Predict on test data
y_pred = model.predict(X_test_flat)

# Visualize results
def visualize_results(images, true_labels, predicted_labels, save_path):
    num_images = images.shape[0]
    fig, axes = plt.subplots(2, num_images // 2, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_images):
        # Convert NDVI values to 0-255 range and display as images
        scaled_image = ((images[i] + 1) * 127.5).astype(np.uint8)
        axes[i].imshow(scaled_image, cmap='gray')
        axes[i].set_title(f'True: {true_labels[i]}\nPredicted: {predicted_labels[i]}')
        axes[i].axis('off')

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    plt.savefig(save_path)  # Save the plot as an image

# Call the visualize_results function with the desired save path
visualize_results(X_test, y_test, y_pred, 'result_plot.png')