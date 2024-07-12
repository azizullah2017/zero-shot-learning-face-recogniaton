import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import matplotlib
matplotlib.use('TkAgg')  # Use 'TkAgg', or another suitable backend
import matplotlib.pyplot as plt
from PIL import Image

# Function to create the base CNN (base network)
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    model = Model(input, x)
    return model

# Function to create the Siamese network
def create_siamese_network(base_network, input_shape, embedding_dim):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Euclidean distance layer
    distance = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)))([processed_a, processed_b])
    
    model = Model([input_a, input_b], distance)
    return model

def load_and_preprocess_image(image_path, target_size):
    img = Image.open(image_path)
    
    # Convert RGBA images to RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img


# Load example images
image_paths = ['example1.png', 'example2.png', 'example3.png']
images = [load_and_preprocess_image(img_path, target_size=(64, 64)) for img_path in image_paths]

# Display the example images
plt.figure(figsize=(8, 4))
for i, img in enumerate(images):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(img)
    plt.title(f'Example {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()


# Simulated attribute embeddings (binary for simplicity)
attributes_embeddings = {
    'example1.png': [1, 1],  # Male, Glasses
    'example2.png': [0, 0],  # Female, No Glasses
    'example3.png': [1, 0]   # Male, No Glasses
}


# Define parameters
input_shape = (64, 64, 3)  # Adjust based on your image dimensions
embedding_dim = len(attributes_embeddings['example1.png'])  # Number of attributes

# Create base network
base_network = create_base_network(input_shape)

# Create Siamese network
siamese_network = create_siamese_network(base_network, input_shape, embedding_dim)
siamese_network.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Prepare training data (pairs of images and their attribute embeddings)
pairs = []
labels = []

for img_path, attributes_embedding in attributes_embeddings.items():
    img = load_and_preprocess_image(img_path, target_size=(64, 64))
    pairs.append((img, img))  # Use the same image as positive pair
    labels.append(1)  # Positive pair label

pairs = np.array(pairs)
labels = np.array(labels)

# Train the model
siamese_network.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=20, batch_size=16)

# Save the model
siamese_network.save('siamese_face_recognition_zsl_model.h5')


# Load a new image for testing
test_image_path = 'example2.png'
test_image = load_and_preprocess_image(test_image_path, target_size=(64, 64))

# Predict similarity with example images
predicted_distances = []

for img_path, attributes_embedding in attributes_embeddings.items():
    img = load_and_preprocess_image(img_path, target_size=(64, 64))
    distance = siamese_network.predict([np.expand_dims(test_image, axis=0), np.expand_dims(img, axis=0)])
    predicted_distances.append(distance[0][0])

# Display predicted distances
print(f"Predicted distances with '{test_image_path}':")
for i, (img_path, _) in enumerate(attributes_embeddings.items()):
    print(f"Distance with '{img_path}': {predicted_distances[i]}")

# Display the test image
plt.figure(figsize=(4, 4))
plt.imshow(test_image)
plt.title('Test Image')
plt.axis('off')
plt.show()

