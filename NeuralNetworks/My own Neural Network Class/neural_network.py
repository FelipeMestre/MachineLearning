import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
except ImportError:  # permite importar el archivo aunque no esté TensorFlow instalado
    tf = None
    mnist = None

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def softmax_derivative(x):
    return x * (1 - x)

def categorical_crossentropy(targets, outputs):
    eps = 1e-12
    losses = []
    for i in range(targets.shape[0]):
        # encuentra la clase correcta (donde targets[i][j] == 1)
        y_i = int(np.argmax(targets[i]))
        p_i = outputs[i][y_i]
        if p_i < eps: p_i = eps
        losses.append(-np.log(p_i))
    return np.array(losses)
    

class NeuralNetwork:
    def __init__(self, input_layer_size, number_of_hidden_layers, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.input_layer = InputLayer(input_layer_size)
        
        self.hidden_layers = []
        prev_size = input_layer_size
        for _ in range(number_of_hidden_layers):
            self.hidden_layers.append(HiddenLayer(prev_size, hidden_layer_size))
            prev_size = hidden_layer_size

        self.output_layer = OutputLayer(prev_size, output_layer_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        previous_layer_activations = self.input_layer.forward(inputs)
        for hidden_layer in self.hidden_layers:
            previous_layer_activations = hidden_layer.forward(previous_layer_activations)
        return self.output_layer.forward(previous_layer_activations)

    def backward(self, targets: np.ndarray) -> np.ndarray:
        error = self.output_layer.backward(targets)
        for hidden_layer in reversed(self.hidden_layers):
            error = hidden_layer.backward(error)
        return error

    def update_weights(self, learning_rate):
        for hidden_layer in self.hidden_layers:
            hidden_layer.update_weights(learning_rate)
        self.output_layer.update_weights(learning_rate)

    def train(self, inputs, targets, epochs, learning_rate, batch_size):
        quantity_of_samples = inputs.shape[0]
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for i in range(0, quantity_of_samples, batch_size):
                batch_inputs_batch = inputs[i:i+batch_size]
                batch_targets_batch = targets[i:i+batch_size]
                output = self.forward(batch_inputs_batch)
                # loss (solo informativa); el backward necesita los targets, no la loss
                _ = categorical_crossentropy(batch_targets_batch, output).mean()
                self.backward(batch_targets_batch)
                self.update_weights(learning_rate)

    def evaluate(self, inputs, targets):
        predictions = self.forward(inputs)
        # targets one-hot: (m, C)
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))

class Layer: 
    def __init__(self, previous_layer_size, size):
        self.size = size
        self.weights = np.random.randn(previous_layer_size, size)
        self.bias = np.random.randn(size)

        self.previous_layer_activations = None
        self.layer_activations = None
        self.layer_outputs = None

        self.output_gradients = None
        self.weights_gradients = np.zeros((previous_layer_size, size))
        self.bias_gradients = np.zeros(size)

    def forward(self, previous_layer_activations: np.ndarray) -> np.ndarray:
        self.previous_layer_activations = previous_layer_activations
        self.layer_activations = np.dot(previous_layer_activations, self.weights) + self.bias
        self.layer_outputs = tanh(self.layer_activations)
        return self.layer_outputs
    
    def backward(self, next_layer_gradients: np.ndarray) -> np.ndarray:
        self.output_gradients = tanh_derivative(self.layer_activations) * next_layer_gradients
        quantity_of_samples = self.output_gradients.shape[0]
        self.weights_gradients = np.dot(self.previous_layer_activations.T, self.output_gradients) / quantity_of_samples
        self.bias_gradients = np.sum(self.output_gradients, axis=0) / quantity_of_samples

        self.previous_layer_gradients = np.dot(self.output_gradients, self.weights.T)
        return self.previous_layer_gradients

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradients
        self.bias -= learning_rate * self.bias_gradients

class InputLayer():
    def __init__(self, size):
        self.size = size

    def forward(self, inputs):
        if inputs.ndim == 1:
            return inputs.reshape(1, -1)
        elif inputs.ndim == 2:
            return inputs
        else:
            raise ValueError(f"Inputs must be 1D or 2D, but got {inputs.ndim}D")
    
    def backward(self, error):
        return error

class HiddenLayer(Layer):
    def __init__(self, previous_layer_size, size):
        super().__init__(previous_layer_size, size)

class OutputLayer(Layer):
    def __init__(self, previous_layer_size, size):
        super().__init__(previous_layer_size, size)
    
    def forward(self, inputs):
        # cachea y aplica softmax sobre Z = XW + b
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        self.previous_layer_activations = inputs
        self.layer_activations = np.dot(inputs, self.weights) + self.bias
        self.layer_outputs = softmax(self.layer_activations)
        return self.layer_outputs
    
    def backward(self, targets: np.ndarray) -> np.ndarray:
        # targets one-hot: (m, C)
        if targets.ndim != 2:
            raise ValueError(
                f"OutputLayer.backward espera targets one-hot con shape (m, C), pero recibió shape {targets.shape}. "
                f"Si tus targets son índices (m,), usa dZ = A; dZ[i, y_i] -= 1."
            )
        quantity_of_samples = targets.shape[0]
        self.output_gradients = self.layer_outputs - targets

        self.weights_gradients = np.dot(self.previous_layer_activations.T, self.output_gradients) / quantity_of_samples
        self.bias_gradients = np.sum(self.output_gradients, axis=0) / quantity_of_samples

        self.previous_layer_gradients = np.dot(self.output_gradients, self.weights.T)
        return self.previous_layer_gradients

if __name__ == "__main__":
    if mnist is None or tf is None:
        raise RuntimeError(
            "TensorFlow no está instalado. Instalalo para correr el ejemplo con MNIST, "
            "o importa este archivo solo para usar la clase NeuralNetwork."
        )

    input_layer_size = 784
    number_of_hidden_layers = 2
    hidden_layer_size = 100
    output_layer_size = 10

    neural_network = NeuralNetwork(input_layer_size, number_of_hidden_layers, hidden_layer_size, output_layer_size)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, input_layer_size) / 255.0
    x_test = x_test.reshape(-1, input_layer_size) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, output_layer_size)
    y_test = tf.keras.utils.to_categorical(y_test, output_layer_size)

    neural_network.train(x_train, y_train, epochs=10, learning_rate=0.01, batch_size=32)

    accuracy = neural_network.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")