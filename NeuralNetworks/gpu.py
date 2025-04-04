##check if GPU is available
## This script checks if a GPU is available for TensorFlow and prints the versions of various libraries.
import sys

import tensorflow.keras as keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow  as tf
import platform
from tensorflow.keras.datasets import mnist

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print(f"SciPy {sp.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
