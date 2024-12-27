import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load the model
model = load_model('real_vs_fake.h5')

# Compile the model (optional but ensures metrics are ready)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Preprocess the input image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)  # Read the image file
    img = tf.image.decode_image(img, channels=3)  # Ensure RGB
    img = tf.image.resize(img, (128, 128))  # Resize to (128, 128)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Display and predict for a given image
def test_image(image_path, expected_label):
    test_img = preprocess_image(image_path)

    # Display the image
    plt.imshow(tf.squeeze(test_img))  # Remove batch dimension for display
    plt.title(f"Testing Image - Expected: {expected_label}")
    plt.axis('off')
    plt.show()

    # Predict using the model
    prediction = model.predict(test_img)[0][0]

    # Interpret and print the result
    if prediction < 0.5:
        print(f"Prediction: Real ({prediction:.2f}) - Expected: {expected_label}")
    else:
        print(f"Prediction: Fake ({1 - prediction:.2f}) - Expected: {expected_label}")


# Test images
real_image_path = "./real.png"  # Replace with the path to your real image
fake_image_path = "./fake.png"  # Replace with the path to your fake image

# Test the real image
test_image(real_image_path, expected_label="Real")

# Test the fake image
test_image(fake_image_path, expected_label="Fake")
