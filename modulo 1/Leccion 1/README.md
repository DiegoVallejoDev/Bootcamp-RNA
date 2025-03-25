# **Lección 1: Introducción a TensorFlow y las Redes Neuronales**

## **1. Introducción**

En esta primera lección, exploraremos qué es TensorFlow y cómo se utiliza para construir modelos de redes neuronales. También daremos un vistazo a los tensores, la estructura de datos fundamental en TensorFlow.

## **2. ¿Qué es TensorFlow?**

TensorFlow es un framework de código abierto desarrollado por Google para construir y entrenar modelos de machine learning y deep learning. Permite definir operaciones matemáticas en grafos computacionales optimizados para CPUs, GPUs y TPUs.

**Características clave:**  
✅ Computación eficiente en paralelo.  
✅ Uso de GPU y TPU para optimización de desempeño.  
✅ API de alto nivel (Keras) para facilidad de uso.  
✅ Soporte para inferencia en dispositivos móviles y edge computing.

## **3. Instalación de TensorFlow**

Para instalar TensorFlow en tu entorno de Python, ejecuta:

```bash
pip install tensorflow
```

Para verificar la instalación, abre un entorno de Python y prueba:

```bash
import tensorflow as tf
print(tf.__version__)
```

Si todo está bien, imprimirá la versión instalada de TensorFlow.

## **4. Tensores: La Base de TensorFlow**

TensorFlow maneja datos en forma de tensores, que son estructuras de datos multidimensionales similares a arrays de NumPy.

**Ejemplo de Creación de Tensores en TensorFlow**

```python
import tensorflow as tf

# Escalar (0D)
tensor_0d = tf.constant(42)

# Vector (1D)
tensor_1d = tf.constant([1, 2, 3])

# Matriz (2D)
tensor_2d = tf.constant([[1, 2], [3, 4]])

# Tensor 3D
tensor_3d = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# Mostrar tensores
print("Escalar:", tensor_0d)
print("Vector:", tensor_1d)
print("Matriz:\n", tensor_2d)
print("Tensor 3D:\n", tensor_3d)
```

## **5. Operaciones Básicas con Tensores**

TensorFlow permite realizar operaciones matemáticas en tensores de manera eficiente.
Ejemplo de Operaciones con Tensores

**Ejemplo de Operaciones con Tensores**

```python
import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

suma = tf.add(a, b)  # Suma
producto = tf.matmul(a, b)  # Multiplicación de matrices
transpuesta = tf.transpose(a)  # Transposición

print("Suma:\n", suma.numpy())
print("Producto de matrices:\n", producto.numpy())
print("Transpuesta de a:\n", transpuesta.numpy())
```

## **6. Representación Matemática de los Tensores**

Un tensor de orden nn es una estructura matemática representada por:

```math
T_{i_1, i_2, ..., i_n}
```

Ejemplos:
```math
T_{i} \text{ (Escalar, 0D)} \\
T_{i,j} \text{ (Vector, 1D)} \\
T_{i,j,k} \text{ (Matriz, 2D)} \\
T_{i,j,k,l} \text{ (Tensor, 3D)}
```
Los tensores pueden ser de diferentes tipos, como enteros, flotantes o booleanos. La forma de un tensor se define por su número de dimensiones y el tamaño de cada dimensión y pueden ser representados como matrices de alto orden, donde cada índice define una nueva dimensión.

## **7. Conclusión**

En esta lección, aprendimos sobre TensorFlow y su estructura base: los tensores. En la siguiente lección, exploraremos las neuronas artificiales y su relación con las funciones de activación.
