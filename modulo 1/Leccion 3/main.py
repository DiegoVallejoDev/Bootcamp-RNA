import numpy as np
import tensorflow as tf

class Perceptron:
    """A simple Perceptron model for binary classification."""

    def __init__(self, input_size: int, learning_rate: float = 0.1, epochs: int = 10) -> None:
        """Initializes the Perceptron with random weights and given parameters.

        Args:
            input_size (int): Number of input features.
            learning_rate (float): Learning rate for weight updates.
            epochs (int): Number of training iterations.
        """
        self.weights: np.ndarray = np.random.rand(input_size + 1)  # +1 for bias
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs

    def activation_function(self, x: float) -> int:
        """Step activation function.

        Args:
            x (float): Weighted sum of inputs.

        Returns:
            int: 1 if x >= 0, else 0.
        """
        return 1 if x >= 0 else 0

    def predict(self, inputs: np.ndarray) -> int:
        """Makes a prediction based on the current weights.

        Args:
            inputs (np.ndarray): Input vector.

        Returns:
            int: Predicted binary output (0 or 1).
        """
        weighted_sum: float = np.dot(inputs, self.weights[1:]) + self.weights[0]  # Bias term
        return self.activation_function(weighted_sum)

    def train(self, training_inputs: np.ndarray, labels: np.ndarray) -> None:
        """Trains the perceptron using the perceptron learning rule.

        Args:
            training_inputs (np.ndarray): Training data inputs.
            labels (np.ndarray): Corresponding binary labels.
        """
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction: int = self.predict(inputs)
                error: int = label - prediction

                # Weight update rule
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error  # Bias update

    def evaluate(self, test_inputs: np.ndarray) -> None:
        """Evaluates the perceptron on test data.

        Args:
            test_inputs (np.ndarray): Input data for evaluation.
        """
        for inputs in test_inputs:
            print(f"Input: {inputs} → Prediction: {self.predict(inputs)}")


if __name__ == "__main__":
    # Training data for AND gate
    and_inputs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_labels: np.ndarray = np.array([0, 0, 0, 1])  # AND truth table

    # Training data for OR gate
    or_inputs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    or_labels: np.ndarray = np.array([0, 1, 1, 1])  # OR truth table

    # Train Perceptron for AND gate
    print("### Training Perceptron for AND Gate ###")
    perceptron_and = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
    perceptron_and.train(and_inputs, and_labels)
    perceptron_and.evaluate(and_inputs)

    # Train Perceptron for OR gate
    print("\n### Training Perceptron for OR Gate ###")
    perceptron_or = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
    perceptron_or.train(or_inputs, or_labels)
    perceptron_or.evaluate(or_inputs)
