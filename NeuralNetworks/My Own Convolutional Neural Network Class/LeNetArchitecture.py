import os
import sys

import numpy as np

try:
    import tensorflow
except ImportError:
    tensorflow = None


current_directory_path = os.path.dirname(__file__)
if current_directory_path not in sys.path:
    sys.path.append(current_directory_path)

neural_network_class_directory_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "My own Neural Network Class")
)
if neural_network_class_directory_path not in sys.path:
    sys.path.append(neural_network_class_directory_path)

from convolutional_neural_network import ConvolutionalLayer
from neural_network import NeuralNetwork


def hyperbolic_tangent(inputs):
    return np.tanh(inputs)


def hyperbolic_tangent_derivative(pre_activation_values):
    return 1.0 - np.tanh(pre_activation_values) ** 2


def one_hot_encode_labels(class_indices, number_of_classes):
    quantity_of_samples = class_indices.shape[0]
    encoded = np.zeros((quantity_of_samples, number_of_classes), dtype="float32")
    encoded[np.arange(quantity_of_samples), class_indices.astype("int64")] = 1.0
    return encoded


def compute_accuracy(predictions, one_hot_encoded_labels):
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(one_hot_encoded_labels, axis=1)
    return float(np.mean(predicted_classes == true_classes))


class HyperbolicTangentActivation:
    def __init__(self):
        self.pre_activation_values = None
        self.outputs = None

    def forward(self, inputs):
        self.pre_activation_values = inputs
        self.outputs = hyperbolic_tangent(inputs)
        return self.outputs

    def backward(self, output_gradients):
        return output_gradients * hyperbolic_tangent_derivative(self.pre_activation_values)


class AveragePooling2D:
    def __init__(self, pool_height=2, pool_width=2, stride=2):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride

        self.previous_layer_outputs = None
        self.output_height = None
        self.output_width = None

    def forward(self, inputs):
        self.previous_layer_outputs = inputs
        batch_size, input_height, input_width, channels = inputs.shape

        self.output_height = int(np.floor((input_height - self.pool_height) / self.stride) + 1)
        self.output_width = int(np.floor((input_width - self.pool_width) / self.stride) + 1)

        outputs = np.zeros((batch_size, self.output_height, self.output_width, channels), dtype=inputs.dtype)

        for batch_index in range(batch_size):
            for output_height_index in range(self.output_height):
                height_start = output_height_index * self.stride
                for output_width_index in range(self.output_width):
                    width_start = output_width_index * self.stride
                    window = inputs[
                        batch_index,
                        height_start : height_start + self.pool_height,
                        width_start : width_start + self.pool_width,
                        :,
                    ]
                    outputs[batch_index, output_height_index, output_width_index, :] = np.mean(
                        window,
                        axis=(0, 1),
                    )
        return outputs

    def backward(self, output_gradients):
        inputs = self.previous_layer_outputs
        batch_size, input_height, input_width, channels = inputs.shape

        input_gradients = np.zeros_like(inputs)
        gradients_per_position = output_gradients / float(self.pool_height * self.pool_width)

        for batch_index in range(batch_size):
            for output_height_index in range(self.output_height):
                height_start = output_height_index * self.stride
                for output_width_index in range(self.output_width):
                    width_start = output_width_index * self.stride
                    input_gradients[
                        batch_index,
                        height_start : height_start + self.pool_height,
                        width_start : width_start + self.pool_width,
                        :,
                    ] += gradients_per_position[batch_index, output_height_index, output_width_index, :]

        return input_gradients


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)

    def backward(self, output_gradients):
        return output_gradients.reshape(self.input_shape)


class LeNet5:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.convolution_1 = ConvolutionalLayer(
            input_channels=1,
            input_width=28,
            input_height=28,
            number_of_filters=6,
            padding="valid",
            stride=1,
            dilation=1,
            kernel_width=5,
            kernel_height=5,
            activation=None,
        )
        self.activation_1 = HyperbolicTangentActivation()
        self.pooling_1 = AveragePooling2D(pool_height=2, pool_width=2, stride=2)

        self.convolution_2 = ConvolutionalLayer(
            input_channels=6,
            input_width=12,
            input_height=12,
            number_of_filters=16,
            padding="valid",
            stride=1,
            dilation=1,
            kernel_width=5,
            kernel_height=5,
            activation=None,
        )
        self.activation_2 = HyperbolicTangentActivation()
        self.pooling_2 = AveragePooling2D(pool_height=2, pool_width=2, stride=2)

        self.convolution_3 = ConvolutionalLayer(
            input_channels=16,
            input_width=4,
            input_height=4,
            number_of_filters=120,
            padding="valid",
            stride=1,
            dilation=1,
            kernel_width=4,
            kernel_height=4,
            activation=None,
        )
        self.activation_3 = HyperbolicTangentActivation()

        self.flatten = Flatten()

        self.fully_connected_network = NeuralNetwork(
            input_layer_size=120,
            number_of_hidden_layers=1,
            hidden_layer_size=84,
            output_layer_size=10,
        )

    def forward(self, inputs):
        activations = self.convolution_1.forward(inputs)
        activations = self.activation_1.forward(activations)
        activations = self.pooling_1.forward(activations)

        activations = self.convolution_2.forward(activations)
        activations = self.activation_2.forward(activations)
        activations = self.pooling_2.forward(activations)

        activations = self.convolution_3.forward(activations)
        activations = self.activation_3.forward(activations)

        activations = self.flatten.forward(activations)
        outputs = self.fully_connected_network.forward(activations)
        return outputs

    def backward(self, targets, debug_one_step=False):
        gradients = self.fully_connected_network.backward(targets, debug=debug_one_step)
        gradients = self.flatten.backward(gradients)

        gradients = self.activation_3.backward(gradients)
        gradients = self.convolution_3.backward(gradients)

        gradients = self.pooling_2.backward(gradients)
        gradients = self.activation_2.backward(gradients)
        gradients = self.convolution_2.backward(gradients)

        gradients = self.pooling_1.backward(gradients)
        gradients = self.activation_1.backward(gradients)
        gradients = self.convolution_1.backward(gradients)
        return gradients

    def update_weights(self, learning_rate):
        self.convolution_1.update_weights(learning_rate)
        self.convolution_2.update_weights(learning_rate)
        self.convolution_3.update_weights(learning_rate)
        self.fully_connected_network.update_weights(learning_rate)

    def train(self, inputs, targets, number_of_epochs, learning_rate, batch_size, maximum_training_samples=None):
        quantity_of_samples = inputs.shape[0]
        if maximum_training_samples is not None:
            quantity_of_samples = min(quantity_of_samples, int(maximum_training_samples))

        for epoch_index in range(number_of_epochs):
            permutation = np.random.permutation(quantity_of_samples)
            inputs_shuffled = inputs[permutation]
            targets_shuffled = targets[permutation]

            for start_index in range(0, quantity_of_samples, batch_size):
                batch_inputs = inputs_shuffled[start_index : start_index + batch_size]
                batch_targets = targets_shuffled[start_index : start_index + batch_size]
                _ = self.forward(batch_inputs)
                self.backward(batch_targets, debug_one_step=False)
                self.update_weights(learning_rate)

    def evaluate_accuracy(self, inputs, targets, batch_size, maximum_test_samples=None):
        quantity_of_samples = inputs.shape[0]
        if maximum_test_samples is not None:
            quantity_of_samples = min(quantity_of_samples, int(maximum_test_samples))

        all_predictions = []
        for start_index in range(0, quantity_of_samples, batch_size):
            batch_inputs = inputs[start_index : start_index + batch_size]
            predictions = self.forward(batch_inputs)
            all_predictions.append(predictions)

        predictions = np.concatenate(all_predictions, axis=0)
        predictions = predictions[:quantity_of_samples]
        targets = targets[:quantity_of_samples]
        return compute_accuracy(predictions, targets)


if __name__ == "__main__":
    if tensorflow is None:
        raise RuntimeError("TensorFlow is required to load MNIST in this script.")

    (training_images, training_labels), (test_images, test_labels) = tensorflow.keras.datasets.mnist.load_data()

    training_images = training_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    training_images = training_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    number_of_classes = 10
    training_labels_one_hot = one_hot_encode_labels(training_labels, number_of_classes)
    test_labels_one_hot = one_hot_encode_labels(test_labels, number_of_classes)

    batch_size = 32
    lenet5_model = LeNet5(batch_size=batch_size)

    number_of_epochs = 10
    learning_rate = 0.001
    maximum_training_samples = 5000
    maximum_test_samples = 5000

    for epoch_index in range(number_of_epochs):
        lenet5_model.train(
            inputs=training_images,
            targets=training_labels_one_hot,
            number_of_epochs=1,
            learning_rate=learning_rate,
            batch_size=batch_size,
            maximum_training_samples=maximum_training_samples,
        )
        accuracy = lenet5_model.evaluate_accuracy(
            inputs=test_images,
            targets=test_labels_one_hot,
            batch_size=batch_size,
            maximum_test_samples=maximum_test_samples,
        )
        print(f"Epoch {epoch_index + 1}/{number_of_epochs} - accuracy: {accuracy:.4f}")

