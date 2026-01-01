import os
import sys

import numpy as np
import tensorflow

neural_network_class_directory_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "My own Neural Network Class")
)
if neural_network_class_directory_path not in sys.path:
    sys.path.append(neural_network_class_directory_path)

from matrix_debugger import MatrixDebugger

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def categorical_crossentropy(predictions, labels):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    losses = -np.sum(labels * np.log(predictions), axis=1)
    return np.mean(losses)

def rectified_linear_unit(inputs):
    return np.maximum(0, inputs)

def rectified_linear_unit_derivative(inputs):
    return (inputs > 0).astype(inputs.dtype)

def reLu(inputs):
    return rectified_linear_unit(inputs)

def reLu_derivative(inputs):
    return rectified_linear_unit_derivative(inputs)

class ConvolutionalNeuralNetwork:
    def __init__(self, batch_size, input_width, input_height, channels):
        self.batch_size = batch_size
        self.input_layer = InputLayer()
        self.convolutional_layers = []
        self.global_average_pooling = None
        self.output_layer = None
    
    def add_convolutional_layer(self, input_channels, input_width, input_height, number_of_filters, 
        padding, stride, dilation, kernel_width, kernel_height, activation="rectified_linear_unit"):
        self.convolutional_layers.append(
            ConvolutionalLayer(
                input_channels=input_channels,
                input_width=input_width,
                input_height=input_height,
                number_of_filters=number_of_filters,
                padding=padding,
                stride=stride,
                dilation=dilation,
                kernel_width=kernel_width,
                kernel_height=kernel_height,
                activation=activation,
            )
        )
    
    def add_global_average_pooling(self):
        self.global_average_pooling = GlobalAveragePooling(self.batch_size)

    def add_output_layer(self, number_of_classes):
        if len(self.convolutional_layers) == 0:
            raise ValueError("You must add at least one convolutional layer before adding the output layer.")
        previous_layer_size = self.convolutional_layers[-1].number_of_filters
        self.output_layer = OutputLayer(number_of_classes, previous_layer_size)
    
    def forward(
        self,
        inputs,
        debug: bool = False,
        debug_max_rows: int = 8,
        debug_max_cols: int = 10,
        debug_precision: int = 4,
    ):
        matrix_debugger = (
            MatrixDebugger(max_rows=debug_max_rows, max_cols=debug_max_cols, precision=debug_precision)
            if debug
            else None
        )

        if matrix_debugger is not None:
            matrix_debugger.title("FORWARD (debug): inputs")
            matrix_debugger.array("X (inputs)", inputs)

        activations = self.input_layer.forward(inputs)

        if matrix_debugger is not None:
            matrix_debugger.title("FORWARD (debug): after input layer")
            matrix_debugger.array("A_0 (activations)", activations)

        for convolutional_layer_index, convolutional_layer in enumerate(self.convolutional_layers):
            if matrix_debugger is not None:
                matrix_debugger.title(
                    f"FORWARD (debug): convolutional layer {convolutional_layer_index}"
                )
                matrix_debugger.array("A_prev (activations)", activations)

            activations = convolutional_layer.forward(activations)

            if matrix_debugger is not None:
                matrix_debugger.array("A (activations)", activations)

        if self.global_average_pooling is not None:
            activations = self.global_average_pooling.forward(activations)

        outputs = self.output_layer.forward(activations)

        if matrix_debugger is not None:
            matrix_debugger.title("FORWARD (debug): output layer")
            matrix_debugger.array("A_L (outputs)", outputs)

        return outputs


    def backward(self, targets):
        gradients = self.output_layer.backward(targets)
        if self.global_average_pooling is not None:
            gradients = self.global_average_pooling.backward(gradients)
        for convolutional_layer in reversed(self.convolutional_layers):
            gradients = convolutional_layer.backward(gradients)
        return gradients

    def update_weights(self, learning_rate):
        for convolutional_layer in self.convolutional_layers:
            convolutional_layer.update_weights(learning_rate)
        self.output_layer.update_weights(learning_rate)

    def train(self, batch_size, inputs, targets, learning_rate):
        quantity_of_samples = inputs.shape[0]
        for initial_index in range(0, quantity_of_samples, batch_size):
            epoch_data = inputs[initial_index: initial_index + batch_size]
            epoch_labels = targets[initial_index: initial_index + batch_size]

            predictions = self.forward(epoch_data)
            loss = categorical_crossentropy(predictions, epoch_labels)
            self.backward(epoch_labels)
            self.update_weights(learning_rate)
        return loss



class InputLayer:
    def __init__(self):
        pass

    def forward(self, inputs):
        return inputs

class ConvolutionalLayer:
    def __init__(
        self, 
        input_channels, 
        input_width, 
        input_height, 
        number_of_filters, 
        padding,
        stride, 
        dilation, 
        kernel_width, 
        kernel_height,
        activation="rectified_linear_unit",
    ):
        self.input_channels = input_channels
        self.input_width = input_width
        self.input_height = input_height
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.number_of_filters = number_of_filters
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.activation = activation

        kernel_height_effective = dilation * (kernel_height - 1) + 1
        kernel_width_effective = dilation * (kernel_width - 1) + 1

        if padding == "same":
            self.output_height = int(np.ceil(input_height / stride))
            self.output_width = int(np.ceil(input_width / stride))
        elif padding == "valid":
            self.output_height = int(np.floor((input_height - kernel_height_effective) / stride) + 1)
            self.output_width = int(np.floor((input_width - kernel_width_effective) / stride) + 1)
        else:
            raise ValueError("padding must be 'same' or 'valid'")

        if padding == "same":
            padding_total_height = max(
                (self.output_height - 1) * self.stride + kernel_height_effective - self.input_height,
                0,
            )
            padding_total_width = max(
                (self.output_width - 1) * self.stride + kernel_width_effective - self.input_width,
                0,
            )
        else:
            padding_total_height = 0
            padding_total_width = 0

        self.padding_top = padding_total_height // 2
        self.padding_bottom = padding_total_height - self.padding_top

        self.padding_left = padding_total_width // 2
        self.padding_right = padding_total_width - self.padding_left

        self.kernels = []
        for _ in range(number_of_filters):
            self.kernels.append(Kernel(kernel_height, kernel_width, input_channels))
        
        self.previous_layer_outputs = None
        self.pre_activation_outputs = None
        self.layer_outputs = None
        self.previous_layer_gradients = None
        self.output_gradients = None
        
    #The shape of inputs must be (batch_size, input_height, input_width, input_channels)
    def forward(self, inputs):
        self.previous_layer_outputs = inputs
        batch_size = inputs.shape[0]
        x_padded = np.pad(inputs, ((0, 0), (self.padding_top, self.padding_bottom), (self.padding_left, self.padding_right), (0, 0)))
        self.pre_activation_outputs = np.zeros((batch_size, self.output_height, self.output_width, self.number_of_filters))

        for batch_index in range(batch_size):
            for filter_index in range(self.number_of_filters):
                kernel = self.kernels[filter_index]

                for h in range(self.output_height):
                    height_start = h * self.stride

                    for w in range(self.output_width):
                        width_start = w * self.stride

                        window = x_padded[
                            batch_index,
                            height_start : height_start + self.dilation * (self.kernel_height - 1) + 1 : self.dilation,
                            width_start : width_start + self.dilation * (self.kernel_width - 1) + 1 : self.dilation,
                            :,
                        ] # Shape: (kernel_height, kernel_width, input_channels)
                        self.pre_activation_outputs[batch_index, h, w, filter_index] = kernel.forward(window)

        if self.activation is None:
            self.layer_outputs = self.pre_activation_outputs
        elif self.activation == "rectified_linear_unit":
            self.layer_outputs = rectified_linear_unit(self.pre_activation_outputs)
        else:
            raise ValueError("Unsupported activation")

        return self.layer_outputs
    
    def backward(self, output_gradient):
        batch_size, _, _, _ = output_gradient.shape

        if self.activation is None:
            self.output_gradients = output_gradient
        elif self.activation == "rectified_linear_unit":
            self.output_gradients = output_gradient * rectified_linear_unit_derivative(self.pre_activation_outputs)
        else:
            raise ValueError("Unsupported activation")

        inputs = self.previous_layer_outputs
        inputs_padded = np.pad(
            inputs,
            ((0, 0), (self.padding_top, self.padding_bottom), (self.padding_left, self.padding_right), (0, 0)),
        )
        input_gradients_padded = np.zeros_like(inputs_padded)

        for kernel in self.kernels:
            kernel.weights_gradient = np.zeros_like(kernel.weights)
            kernel.bias_gradient = 0.0

        for batch_index in range(batch_size):
            for output_height_index in range(self.output_height):
                height_start = output_height_index * self.stride
                for output_width_index in range(self.output_width):
                    width_start = output_width_index * self.stride

                    input_window = inputs_padded[
                        batch_index,
                        height_start : height_start + self.dilation * (self.kernel_height - 1) + 1 : self.dilation,
                        width_start : width_start + self.dilation * (self.kernel_width - 1) + 1 : self.dilation,
                        :,
                    ]

                    for filter_index, kernel in enumerate(self.kernels):
                        upstream_gradient = self.output_gradients[batch_index, output_height_index, output_width_index, filter_index]
                        kernel.weights_gradient += input_window * upstream_gradient
                        kernel.bias_gradient += float(upstream_gradient)

                        input_gradients_padded[
                            batch_index,
                            height_start : height_start + self.dilation * (self.kernel_height - 1) + 1 : self.dilation,
                            width_start : width_start + self.dilation * (self.kernel_width - 1) + 1 : self.dilation,
                            :,
                        ] += kernel.weights * upstream_gradient

        for kernel in self.kernels:
            kernel.weights_gradient = kernel.weights_gradient / batch_size
            kernel.bias_gradient = kernel.bias_gradient / batch_size

        input_gradients = input_gradients_padded[
            :,
            self.padding_top : input_gradients_padded.shape[1] - self.padding_bottom,
            self.padding_left : input_gradients_padded.shape[2] - self.padding_right,
            :,
        ]
        self.previous_layer_gradients = input_gradients
        return self.previous_layer_gradients
    
    def update_weights(self, learning_rate):
        for kernel in self.kernels:
            kernel.update_weights(learning_rate)
    
    def update_bias(self, learning_rate):
        for kernel in self.kernels:
            kernel.update_bias(learning_rate)
    
class Kernel:
    def __init__(self, kernel_height, kernel_width, input_channels) -> None:
        self.weights = np.random.randn(kernel_height, kernel_width, input_channels).astype("float32")
        self.bias = float(np.random.randn())
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = 0.0
        
    def forward(self, window):
        return np.sum(window * self.weights) + self.bias #Element to element multiplication sum
    
    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradient

    def update_bias(self, learning_rate):
        self.bias -= learning_rate * self.bias_gradient

class GlobalAveragePooling:
    def __init__(self, batch_size, keep_dimensions: bool = False):
        """
        - inputs shape:  (batch_size, height, width, channels)
        - outputs shape: (batch_size, channels) si keep_dimensions=False
                        (batch_size, 1, 1, channels) si keep_dimensions=True
        """
        self.keep_dimensions = keep_dimensions
        self._input_shape = None

    def forward(self, inputs):
        # inputs: (batch_size, height, width, channels)
        self._input_shape = inputs.shape
        return inputs.mean(axis=(1, 2), keepdims=self.keep_dimensions)

    def backward(self, output_gradients):
        # Output_gradients has shape (batch_size, number_of_classes)
        batch_size, height, width, channels = self._input_shape
        output_gradients_expanded = output_gradients.reshape(batch_size, 1, 1, channels)

        # Each channel means over height*width => gradients are speared uniformly
        input_gradients_per_position = output_gradients_expanded / (height * width)
        input_gradients = np.broadcast_to(
            input_gradients_per_position,
            (batch_size, height, width, channels),
        ).copy()
        return input_gradients


class OutputLayer: 
    def __init__(self, number_of_classes, previous_layer_size):
        self.number_of_classes = number_of_classes
        self.previous_layer_size = previous_layer_size
        self.weights = np.random.randn(previous_layer_size, number_of_classes).astype("float32") * 0.01
        self.bias = np.zeros((1, number_of_classes), dtype="float32")
        
        self.weights_gradients = None
        self.bias_gradients = None

        self.previous_layer_outputs = None
        self.layer_outputs = None
        self.previous_layer_gradients = None
        
    def forward(self, inputs):
        if inputs.ndim == 4:
            inputs = inputs.reshape(inputs.shape[0], -1)
        self.previous_layer_outputs = inputs
        logits = np.dot(inputs, self.weights) + self.bias
        self.layer_outputs = softmax(logits)
        return self.layer_outputs
    
    def backward(self, targets):
        if targets.ndim != 2:
            raise ValueError("targets must be one-hot encoded with shape (batch_size, number_of_classes)")

        batch_size = targets.shape[0]
        output_gradients = (self.layer_outputs - targets) / batch_size

        self.weights_gradients = np.dot(self.previous_layer_outputs.T, output_gradients)
        self.bias_gradients = np.sum(output_gradients, axis=0, keepdims=True)
        self.previous_layer_gradients = np.dot(output_gradients, self.weights.T)
        return self.previous_layer_gradients

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradients
        self.bias -= learning_rate * self.bias_gradients

if __name__ == "__main__":
    np.random.seed(7)

    batch_size = 4
    input_width = 28
    input_height = 28
    input_channels = 1
    number_of_classes = 10

    stride = 1
    dilation = 1

    convolutional_layer_0_number_of_filters = 16
    convolutional_layer_0_kernel_width = 3
    convolutional_layer_0_kernel_height = 3

    convolutional_layer_1_number_of_filters = 32
    convolutional_layer_1_kernel_width = 5
    convolutional_layer_1_kernel_height = 5

    convolutional_layer_2_number_of_filters = 64
    convolutional_layer_2_kernel_width = 3
    convolutional_layer_2_kernel_height = 3

    convolutional_layers = []

    convolutional_layer_0 = ConvolutionalLayer(
        input_channels=input_channels,
        input_width=input_width,
        input_height=input_height,
        number_of_filters=convolutional_layer_0_number_of_filters,
        padding="same",
        stride=stride,
        dilation=dilation,
        kernel_width=convolutional_layer_0_kernel_width,
        kernel_height=convolutional_layer_0_kernel_height,
    )
    convolutional_layers.append(convolutional_layer_0)

    convolutional_layer_1 = ConvolutionalLayer(
        input_channels=convolutional_layer_0_number_of_filters,
        input_width=convolutional_layer_0.output_width,
        input_height=convolutional_layer_0.output_height,
        number_of_filters=convolutional_layer_1_number_of_filters,
        padding="same",
        stride=stride,
        dilation=dilation,
        kernel_width=convolutional_layer_1_kernel_width,
        kernel_height=convolutional_layer_1_kernel_height,
    )
    convolutional_layers.append(convolutional_layer_1)

    convolutional_layer_2 = ConvolutionalLayer(
        input_channels=convolutional_layer_1_number_of_filters,
        input_width=convolutional_layer_1.output_width,
        input_height=convolutional_layer_1.output_height,
        number_of_filters=convolutional_layer_2_number_of_filters,
        padding="same",
        stride=stride,
        dilation=dilation,
        kernel_width=convolutional_layer_2_kernel_width,
        kernel_height=convolutional_layer_2_kernel_height,
    )
    convolutional_layers.append(convolutional_layer_2)

    global_average_pooling = GlobalAveragePooling(batch_size=batch_size)
    output_layer = OutputLayer(
        number_of_classes=number_of_classes,
        previous_layer_size=convolutional_layer_2_number_of_filters,
    )

    inputs = np.random.randn(batch_size, input_height, input_width, input_channels).astype("float32")

    print("Input shape:", inputs.shape)

    activations = inputs
    for layer_index, convolutional_layer in enumerate(convolutional_layers):
        activations = convolutional_layer.forward(activations)
        print(f"After convolutional layer {layer_index} shape:", activations.shape)

    pooled = global_average_pooling.forward(activations)
    print("After global average pooling shape:", pooled.shape)

    matrix_debugger = MatrixDebugger(max_rows=4, max_cols=8, precision=4)
    matrix_debugger.title("FORWARD (debug): inputs")
    matrix_debugger.array("X (inputs)", inputs)
    matrix_debugger.title("FORWARD (debug): after last convolutional layer")
    matrix_debugger.array("A_conv_last (activations)", activations)
    matrix_debugger.title("FORWARD (debug): after global average pooling")
    matrix_debugger.array("A_gap (pooled)", pooled)

    probabilities = output_layer.forward(pooled)
    matrix_debugger.title("FORWARD (debug): output layer")
    matrix_debugger.array("A_L (probabilities)", probabilities)
    print("Output probabilities shape:", probabilities.shape)

    def one_hot_encode_labels(class_indices, number_of_classes):
        quantity_of_samples = class_indices.shape[0]
        encoded = np.zeros((quantity_of_samples, number_of_classes), dtype="float32")
        encoded[np.arange(quantity_of_samples), class_indices.astype("int64")] = 1.0
        return encoded

    def compute_accuracy(predictions, one_hot_encoded_labels):
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(one_hot_encoded_labels, axis=1)
        return float(np.mean(predicted_classes == true_classes))

    def train_and_evaluate_mnist():

        (training_images, training_labels), (test_images, test_labels) = tensorflow.keras.datasets.mnist.load_data()

        training_images = training_images.astype("float32") / 255.0
        test_images = test_images.astype("float32") / 255.0

        training_images = training_images.reshape(-1, 28, 28, 1)
        test_images = test_images.reshape(-1, 28, 28, 1)

        number_of_classes = 10
        training_labels_one_hot = one_hot_encode_labels(training_labels, number_of_classes)
        test_labels_one_hot = one_hot_encode_labels(test_labels, number_of_classes)

        maximum_training_samples = 2048
        maximum_test_samples = 512
        training_images = training_images[:maximum_training_samples]
        training_labels_one_hot = training_labels_one_hot[:maximum_training_samples]
        test_images = test_images[:maximum_test_samples]
        test_labels_one_hot = test_labels_one_hot[:maximum_test_samples]

        batch_size = 32
        convolutional_neural_network = ConvolutionalNeuralNetwork(
            batch_size=batch_size,
            input_width=28,
            input_height=28,
            channels=1,
        )
        convolutional_neural_network.add_convolutional_layer(
            input_channels=1,
            input_width=28,
            input_height=28,
            number_of_filters=8,
            padding="same",
            stride=1,
            dilation=1,
            kernel_width=3,
            kernel_height=3,
            activation="rectified_linear_unit",
        )
        convolutional_neural_network.add_global_average_pooling()
        convolutional_neural_network.add_output_layer(number_of_classes=number_of_classes)

        number_of_epochs = 3
        learning_rate = 0.01

        for epoch_index in range(number_of_epochs):
            permutation = np.random.permutation(training_images.shape[0])
            training_images_shuffled = training_images[permutation]
            training_labels_shuffled = training_labels_one_hot[permutation]

            last_loss = None
            for start_index in range(0, training_images_shuffled.shape[0], batch_size):
                batch_images = training_images_shuffled[start_index : start_index + batch_size]
                batch_labels = training_labels_shuffled[start_index : start_index + batch_size]
                last_loss = convolutional_neural_network.train(
                    batch_size=batch_size,
                    inputs=batch_images,
                    targets=batch_labels,
                    learning_rate=learning_rate,
                )

            predictions = convolutional_neural_network.forward(test_images)
            accuracy = compute_accuracy(predictions, test_labels_one_hot)
            print(f"Epoch {epoch_index + 1}/{number_of_epochs} - loss: {last_loss:.6f} - accuracy: {accuracy:.4f}")

    train_and_evaluate_mnist()