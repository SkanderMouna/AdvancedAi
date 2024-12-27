import os
import sklearn.model_selection as ms
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dropout, Conv2D, Flatten, MaxPooling2D, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import models
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths to fake and real picture datasets
fake_pictures_path = "./fakeFaces/training_fake"
real_pictures_path = "./fakeFaces/training_real"
newsize = (128, 128)

# Preprocessing function for images
def preprocessimages(images_dir, newsize):
    filenames = os.listdir(images_dir)
    tmp = []
    for filename in filenames:
        file_path = os.path.join(images_dir, filename)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)  # Convert to RGB
        img = tf.image.resize(img, newsize)  # Resize to (128, 128)
        img = img / 255.0  # Normalize pixel values
        tmp.append(img)
    return tf.stack(tmp)

# Preprocess the fake and real pictures
fake_pictures = preprocessimages(fake_pictures_path, newsize)
real_pictures = preprocessimages(real_pictures_path, newsize)

# Generate labels
fake_labels = tf.zeros(tf.shape(fake_pictures)[0], dtype=tf.int32)  # Label fake as 0
real_labels = tf.ones(tf.shape(real_pictures)[0], dtype=tf.int32)   # Label real as 1

# Combine fake and real images and labels
X = tf.concat([fake_pictures, real_pictures], axis=0).numpy().astype('float32')
y = tf.concat([fake_labels, real_labels], axis=0).numpy().astype('int32')

# Split the dataset into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = ms.train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = ms.train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Display dataset information
print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)
print("y_train shape:", y_train.shape, "dtype:", y_train.dtype)
print("X_val shape:", X_val.shape, "dtype:", X_val.dtype)
print("y_val shape:", y_val.shape, "dtype:", y_val.dtype)

# Define the CNN model
model = models.Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3), kernel_regularizer=l2(0.001)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

print(model.summary())

# Compile the model
optimizer = Adam(learning_rate=0.0001)  # Lower learning rate for fine-tuning
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Training callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
try:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )
except Exception as e:
    print("Error during model.fit:", str(e))

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=32)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Confusion matrix and classification report
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype("int32")
conf_matrix = confusion_matrix(y_test, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)
class_report = classification_report(y_test, predicted_classes, target_names=['Fake', 'Real'])
print("Classification Report:")
print(class_report)

# Save the fine-tuned model
model.save("./fakeFaces/real_vs_fake.h5")
print("Model saved to './fakeFaces/real_vs_fake.h5'")
