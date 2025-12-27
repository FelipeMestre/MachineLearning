import numpy as np

def softmax(x):
    x = x - np.max(x, axis=0, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=0, keepdims=True)

def categorical_crossentropy(predictions, labels):
    return -np.sum(labels * np.log(predictions))

class ConvolutionalNeuralNetwork:
    def __init__(self, batch_size, input_width, input_height, channels):
        self.input_layer = InputLayer(input_width, input_height, channels)
        self.convolutional_layers = []
    
    def add_convolutional_layer(self, input_channels, input_width, input_height, number_of_filters, 
        padding, stride, dilation, kernel_width, kernel_height):
        self.convolutional_layers.append(ConvolutionalLayer(input_channels, input_width, input_height, 
            number_of_filters, padding, stride, dilation, kernel_width, kernel_height, self.batch_size))
    
    def add_global_average_pooling(self):
        self.global_average_pooling = GlobalAveragePooling(self.batch_size)

    def add_output_layer(self, number_of_classes):
        self.output_layer = OutputLayer(number_of_classes, self.batch_size)
    
    def forward(self, inputs):
        inputs = self.input_layer.forward(inputs)
        for convolutional_layer in self.convolutional_layers:
            inputs = convolutional_layer.forward(inputs)
        return self.output_layer.forward(inputs)


    def train(self, batch_size, inputs, targets, learning_rate):
        quantity_of_samples = inputs.shape[0]
        for initial_index in range(0, quantity_of_samples, batch_size):
            epoch_data = inputs[initial_index: initial_index + batch_size]
            epoch_labels = targets[initial_index: initial_index + batch_size]

            predictions = self.forward(epoch_data)
            loss = categorical_crossentropy(predictions, epoch_labels)
            gradient = self.output_layer.backward(loss)
            
            for convolutional_layer in reversed(self.convolutional_layers):
                gradient = convolutional_layer.backward(gradient)
                
        for convolutional_layer in self.convolutional_layers:
            convolutional_layer.update_weights(learning_rate)
        self.output_layer.update_weights(learning_rate)



class InputLayer:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
    
    def forward(self, inputs):
        return inputs

class ConvolutionalLayer:
    def __init__(
        self, 
        input_channels, 
        input_width, 
        input_height, 
        number_of_filters, 
        stride, 
        dilation, 
        kernel_width, 
        kernel_height
    ):
        self.input_channels = input_channels
        self.input_width = input_width
        self.input_height = input_height
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.number_of_filters = number_of_filters
        self.stride = stride
        self.dilation = dilation

        kernel_height_effective = dilation * (kernel_height - 1) + 1
        kernel_width_effective = dilation * (kernel_width - 1) + 1

        self.output_height = int(np.ceil(input_height / stride))
        self.output_width = int(np.ceil(input_width  / stride))

        padding_total_height = max((self.output_height - 1) * self.stride + kernel_height_effective - self.input_height, 0)
        padding_total_width  = max((self.output_width  - 1) * self.stride + kernel_width_effective  - self.input_width, 0)

        self.padding_top = padding_total_height // 2
        self.padding_bottom = padding_total_height - self.padding_top

        self.padding_left = padding_total_width // 2
        self.padding_right = padding_total_width - self.padding_left

        self.kernels = []
        for _ in range(number_of_filters):
            self.kernels.append(Kernel(kernel_height, kernel_width, input_channels, number_of_filters))
        
        self.previous_layer_outputs = None
        self.layer_outputs = None
        self.output_gradient = None
        self.previous_layer_gradients = None
        
    #The shape of inputs must be (batch_size, input_height, input_width, input_channels)
    def forward(self, inputs):
        self.previous_layer_outputs = inputs
        batch_size = inputs.shape[0]
        x_padded = np.pad(inputs, ((0, 0), (self.padding_top, self.padding_bottom), (self.padding_left, self.padding_right), (0, 0)))
        self.layer_outputs = np.zeros((batch_size, self.output_height, self.output_width, self.number_of_filters))

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
                        self.layer_outputs[batch_index, h, w, filter_index] = kernel.forward(window)
        return self.layer_outputs
    
    def backward(self, inputs, output_gradient):
        self.previous_layer_outputs = inputs
        self.layer_outputs = output_gradient
        return self.previous_layer_outputs
    
    def update_weights(self, learning_rate):
        for i in range(self.number_of_filters):
            for j in range(self.input_height - self.kernel_height + 1):
                for k in range(self.input_width - self.kernel_width + 1):
                    self.kernels[i].update_weights(learning_rate)
    
    def update_bias(self, learning_rate):
        for i in range(self.number_of_filters):
            self.kernels[i].update_bias(learning_rate)
    
class Kernel:
    def __init__(self, kernel_height, kernel_width, input_channels, output_channels) -> None:
        self.weights = np.random.randn(kernel_height, kernel_width, input_channels, output_channels)
        self.bias = np.random.randn(output_channels)
        
    def forward(self, window):
        return np.sum(window * self.weights) + self.bias # Element to element multiplication sum
    
    def backward(self, input, output_gradient):
        self.weights_gradient = np.dot(input.T, output_gradient)
        self.bias_gradient = np.sum(output_gradient)
        return np.dot(output_gradient, self.weights.T)

class GlobalAveragePooling:
    def __init__(self, batch_size, keep_dimensions: bool = False):
        """
        Global Average Pooling 2D para inputs en formato NHWC:
        - inputs shape:  (batch_size, height, width, channels)
        - outputs shape: (batch_size, channels) si keep_dimensions=False
                        (batch_size, 1, 1, channels) si keep_dimensions=True
        """
        self.keep_dimensions = keep_dimensions
        self._input_shape = None

    def forward(self, inputs):
        # inputs: (batch_size, height, width, channels)
        if inputs.ndim != 4:
            raise ValueError(
                f"GlobalAveragePooling.forward espera inputs con shape (batch_size, height, width, channels), "
                f"pero recibió {inputs.shape}."
            )
        self._input_shape = inputs.shape
        return inputs.mean(axis=(1, 2), keepdims=self.keep_dimensions)

    def backward(self, output_gradients):
        if self._input_shape is None:
            raise RuntimeError("GlobalAveragePooling.backward llamado antes de forward.")

        batch_size, height, width, channels = self._input_shape

        # Aceptamos gradiente como:
        # - (batch_size, channels)
        # - (batch_size, 1, 1, channels) si keep_dimensions=True aguas arriba
        if output_gradients.ndim == 2:
            if output_gradients.shape != (batch_size, channels):
                raise ValueError(
                    f"GlobalAveragePooling.backward recibió output_gradients con shape {output_gradients.shape}, "
                    f"pero esperaba {(batch_size, channels)}."
                )
            output_gradients_expanded = output_gradients.reshape(batch_size, 1, 1, channels)
        elif output_gradients.ndim == 4:
            if output_gradients.shape != (batch_size, 1, 1, channels):
                raise ValueError(
                    f"GlobalAveragePooling.backward recibió output_gradients con shape {output_gradients.shape}, "
                    f"pero esperaba {(batch_size, 1, 1, channels)}."
                )
            output_gradients_expanded = output_gradients
        else:
            raise ValueError(
                f"GlobalAveragePooling.backward espera output_gradients 2D o 4D, pero recibió {output_gradients.shape}."
            )

        # Cada canal se promedia sobre height*width => el gradiente se reparte uniformemente.
        input_gradients_per_position = output_gradients_expanded / (height * width)
        input_gradients = np.broadcast_to(
            input_gradients_per_position,
            (batch_size, height, width, channels),
        ).copy()
        return input_gradients


class OutputLayer: 
    def __init__(self, number_of_classes, previous_layer_size, batch_size):
        self.batch_size = batch_size
        self.weights = np.zeros((previous_layer_size, number_of_classes), dtype="float32")
        self.bias = np.zeros(number_of_classes, dtype="float32")
        
        self.previous_layer_outputs = None
        self.layer_outputs = None
        self.output_gradient = None
        self.previous_layer_gradients = None
        
    def forward(self, inputs):
        self.previous_layer_outputs = inputs
        self.layer_outputs = softmax(np.dot(inputs, self.weights) + self.bias)
        return self.layer_outputs
    
    def backward(self, predictions, labels):
        for label in labels:
            error = predictions - labels

            
            for convolutional_layer in reversed[ConvolutionalLayer](self.convolutional_layers):
                convolutional_layer.backward(predictions, error)

if __name__ == "__main__":
    # Forward-only sanity test (NHWC): create layers manually to match the current class signatures above.
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
        stride=stride,
        dilation=dilation,
        kernel_width=convolutional_layer_0_kernel_width,
        kernel_height=convolutional_layer_0_kernel_height,
    )
    # Adjust each Kernel to behave as "one kernel -> one filter output" without changing class code above.
    for filter_index, kernel in enumerate(convolutional_layer_0.kernels):
        kernel.weights = kernel.weights[..., filter_index]  # (kernel_height, kernel_width, input_channels)
        kernel.bias = float(kernel.bias[filter_index])
    convolutional_layers.append(convolutional_layer_0)

    convolutional_layer_1 = ConvolutionalLayer(
        input_channels=convolutional_layer_0_number_of_filters,
        input_width=convolutional_layer_0.output_width,
        input_height=convolutional_layer_0.output_height,
        number_of_filters=convolutional_layer_1_number_of_filters,
        stride=stride,
        dilation=dilation,
        kernel_width=convolutional_layer_1_kernel_width,
        kernel_height=convolutional_layer_1_kernel_height,
    )
    for filter_index, kernel in enumerate(convolutional_layer_1.kernels):
        kernel.weights = kernel.weights[..., filter_index]
        kernel.bias = float(kernel.bias[filter_index])
    convolutional_layers.append(convolutional_layer_1)

    convolutional_layer_2 = ConvolutionalLayer(
        input_channels=convolutional_layer_1_number_of_filters,
        input_width=convolutional_layer_1.output_width,
        input_height=convolutional_layer_1.output_height,
        number_of_filters=convolutional_layer_2_number_of_filters,
        stride=stride,
        dilation=dilation,
        kernel_width=convolutional_layer_2_kernel_width,
        kernel_height=convolutional_layer_2_kernel_height,
    )
    for filter_index, kernel in enumerate(convolutional_layer_2.kernels):
        kernel.weights = kernel.weights[..., filter_index]
        kernel.bias = float(kernel.bias[filter_index])
    convolutional_layers.append(convolutional_layer_2)

    global_average_pooling = GlobalAveragePooling(batch_size=batch_size)
    output_layer = OutputLayer(
        number_of_classes=number_of_classes,
        previous_layer_size=convolutional_layer_2_number_of_filters,
        batch_size=batch_size,
    )

    inputs = np.random.randn(batch_size, input_height, input_width, input_channels).astype("float32")

    print("Input shape:", inputs.shape)

    activations = inputs
    for layer_index, convolutional_layer in enumerate(convolutional_layers):
        activations = convolutional_layer.forward(activations)
        print(f"After convolutional layer {layer_index} shape:", activations.shape)

    pooled = global_average_pooling.forward(activations)
    print("After global average pooling shape:", pooled.shape)

    probabilities = output_layer.forward(pooled)
    print("Output probabilities shape:", probabilities.shape)