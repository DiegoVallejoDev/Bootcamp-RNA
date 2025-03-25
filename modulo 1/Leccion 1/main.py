import tensorflow as tf
import numpy as np

def create_tensors() -> None:
    """Creates and prints different types of tensors."""
    
    # Scalar (0D)
    tensor_0d: tf.Tensor = tf.constant(42, dtype=tf.int32)

    # Vector (1D)
    tensor_1d: tf.Tensor = tf.constant([1, 2, 3], dtype=tf.int32)

    # Matrix (2D)
    tensor_2d: tf.Tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)

    # 3D Tensor
    tensor_3d: tf.Tensor = tf.constant(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.int32
    )

    # Print tensors
    print("Scalar:", tensor_0d.numpy())
    print("Vector:", tensor_1d.numpy())
    print("Matrix:\n", tensor_2d.numpy())
    print("3D Tensor:\n", tensor_3d.numpy())

def tensor_operations() -> None:
    """Performs basic operations on tensors and prints the results."""
    
    a: tf.Tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    b: tf.Tensor = tf.constant([[5, 6], [7, 8]], dtype=tf.int32)

    sum_result: tf.Tensor = tf.add(a, b)  # Addition
    product_result: tf.Tensor = tf.matmul(a, b)  # Matrix multiplication
    transpose_result: tf.Tensor = tf.transpose(a)  # Transpose

    # Print results
    print("Addition:\n", sum_result.numpy())
    print("Matrix Multiplication:\n", product_result.numpy())
    print("Transpose:\n", transpose_result.numpy())

if __name__ == "__main__":
    print("### Tensor Creation ###")
    create_tensors()
    
    print("\n### Tensor Operations ###")
    tensor_operations()
