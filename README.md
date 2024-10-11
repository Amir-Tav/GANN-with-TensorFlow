# FashionGAN: Generative Adversarial Network for Fashion Images

This repository demonstrates how to build, train, and evaluate a Generative Adversarial Network (GAN) for generating fashion images. The project covers all essential steps, from model building to training and evaluation.

---

## 1. Setting Up the Environment

Install the required libraries and dependencies:

```bash
pip install tensorflow matplotlib
```

---

## 2. Building the Generator and Discriminator

The generator creates new images from random noise, while the discriminator evaluates the authenticity of generated images. Key parts of the generator and discriminator are outlined below:

- **Generator Model:**
  - A sequential model with dense and convolutional layers.

- **Discriminator Model:**
  - Convolutional layers with Leaky ReLU activations and dropout for regularization.

```python
def build_generator():
    model = Sequential()
    # Add layers to the model
    return model

def build_discriminator():
    model = Sequential()
    # Add layers to the model
    return model
```

---

## 3. Training the GAN

The GAN is trained using a custom training loop, where both the generator and discriminator are updated iteratively:

```python
for epoch in range(epochs):
    # Train discriminator and generator
```

---

## 4. Monitoring and Saving the Model

Utilize callbacks to monitor training progress and save generated images:

```python
class ModelMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save generated images
```

---

## 5. Testing the Generator

After training, the generator can produce new images based on learned features:

```python
generator.load_weights('path_to_weights')
generated_images = generator.predict(tf.random.normal((16, 128, 1)))
```

---

## Conclusion

This project successfully demonstrates how to:
* Build a GAN with TensorFlow and Keras.
* Train the model and visualize generated images.
* Evaluate the performance of the generator and discriminator.
*also note: the model is only trained with 5 epoch, but the outcome is still impresive!
---
