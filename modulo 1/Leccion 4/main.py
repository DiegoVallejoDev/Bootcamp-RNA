import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import generate_data

class Adaline(tf.keras.Model):
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        """Initializes the Adaline model with one linear neuron."""
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=1, activation=None, use_bias=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs a forward pass through the Adaline model."""
        return self.dense(inputs)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 50) -> None:
        """Trains the Adaline model using Mean Squared Error and Gradient Descent."""
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                predictions: tf.Tensor = self.call(x_train)
                loss: tf.Tensor = tf.reduce_mean(tf.square(y_train - predictions))  # MSE Loss

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

# Load dataset
x_train, y_train = generate_data(num_samples=100)

# Create and train Adaline model
adaline_model = Adaline(input_dim=1, learning_rate=0.01)
adaline_model.train(x_train, y_train, epochs=100)

# Plot results
y_pred = adaline_model.call(x_train).numpy()
plt.scatter(x_train, y_train, label="Actual Data")
plt.plot(x_train, y_pred, color="red", label="Adaline Predictions")
plt.legend()
plt.title("Adaline Regression")
plt.show()
