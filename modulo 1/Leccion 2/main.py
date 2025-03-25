import tensorflow as tf
import numpy as np

def step_function(x: tf.Tensor) -> tf.Tensor:
    """Step activation function."""
    return tf.where(x >= 0, 1.0, 0.0)

def mcculloch_pitts_neuron(inputs: tf.Tensor, weights: tf.Tensor, threshold: float) -> tf.Tensor:
    """Simulates a McCulloch-Pitts neuron."""
    weighted_sum: tf.Tensor = tf.matmul(inputs, weights) + threshold
    return step_function(weighted_sum)

def logic_gates_demo() -> None:
    """Demonstrates AND and OR gates using McCulloch-Pitts neurons."""
    
    # Logical inputs
    inputs: tf.Tensor = tf.constant([[0,0], [0,1], [1,0], [1,1]], dtype=tf.float32)

    # AND Gate Parameters
    and_weights: tf.Tensor = tf.constant([[1.0], [1.0]], dtype=tf.float32)
    and_threshold: float = -1.5

    # OR Gate Parameters
    or_weights: tf.Tensor = tf.constant([[1.0], [1.0]], dtype=tf.float32)
    or_threshold: float = -0.5

    # Compute outputs
    and_output: tf.Tensor = mcculloch_pitts_neuron(inputs, and_weights, and_threshold)
    or_output: tf.Tensor = mcculloch_pitts_neuron(inputs, or_weights, or_threshold)

    # Print results
    print("AND Gate Output:\n", and_output.numpy())
    print("OR Gate Output:\n", or_output.numpy())

if __name__ == "__main__":
    logic_gates_demo()