import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
except ImportError:  # permite importar el archivo aunque no esté TensorFlow instalado
    tf = None
    mnist = None

from matrix_debugger import _MatrixDebugger

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

    def backward(
        self,
        targets: np.ndarray,
        debug: bool = False,
        debug_max_rows: int = 8,
        debug_max_cols: int = 10,
        debug_precision: int = 4,
    ) -> np.ndarray:
        dbg = _MatrixDebugger(max_rows=debug_max_rows, max_cols=debug_max_cols, precision=debug_precision) if debug else None

        if dbg is not None:
            dbg.title("BACKWARD (debug): capa de salida")
            dbg.array("Y (targets)", targets)
            dbg.array("A_L (outputs)", self.output_layer.layer_outputs)

        error = self.output_layer.backward(targets)

        if dbg is not None:
            dbg.array("Z_L (pre-activation)", self.output_layer.layer_activations)
            dbg.array("dZ_L (output_gradients)", self.output_layer.output_gradients)
            dbg.array("dW_L (weights_gradients)", self.output_layer.weights_gradients)
            dbg.array("db_L (bias_gradients)", self.output_layer.bias_gradients)
            dbg.array("dA_{L-1} (previous_layer_gradients)", self.output_layer.previous_layer_gradients)

        # hidden layers: recorremos de atrás hacia delante
        total_hidden = len(self.hidden_layers)
        for k, hidden_layer in enumerate(reversed(self.hidden_layers), start=1):
            incoming = error
            if dbg is not None:
                layer_index = total_hidden - k + 1
                dbg.title(f"BACKWARD (debug): hidden layer {layer_index}")
                dbg.array("A_prev (cached)", hidden_layer.previous_layer_activations)
                dbg.array("Z (pre-activation)", hidden_layer.layer_activations)
                dbg.array("A (activation)", hidden_layer.layer_outputs)
                dbg.array("dA_next (incoming gradient)", incoming)

            error = hidden_layer.backward(incoming)

            if dbg is not None:
                dbg.array("dZ (output_gradients)", hidden_layer.output_gradients)
                dbg.array("dW (weights_gradients)", hidden_layer.weights_gradients)
                dbg.array("db (bias_gradients)", hidden_layer.bias_gradients)
                dbg.array("dA_prev (previous_layer_gradients)", hidden_layer.previous_layer_gradients)

        return error

    def update_weights(self, learning_rate):
        for hidden_layer in self.hidden_layers:
            hidden_layer.update_weights(learning_rate)
        self.output_layer.update_weights(learning_rate)

    def train(self, inputs, targets, epochs, learning_rate, batch_size, debug_one_backward_step: bool = False):
        quantity_of_samples = inputs.shape[0]
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for i in range(0, quantity_of_samples, batch_size):
                batch_inputs_batch = inputs[i:i+batch_size]
                batch_targets_batch = targets[i:i+batch_size]
                # output shape(batch_size, possible_classes)
                output = self.forward(batch_inputs_batch)
                # loss (solo informativa); el backward necesita los targets, no la loss
                _ = categorical_crossentropy(batch_targets_batch, output).mean()
                self.backward(batch_targets_batch, debug=debug_one_backward_step)
                self.update_weights(learning_rate)
                if debug_one_backward_step:
                    debug_one_backward_step = False

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

        #dLoss/dpre_activation
        self.output_gradients = self.layer_outputs - targets

        #dLoss/dWeights
        self.weights_gradients = np.dot(self.previous_layer_activations.T, self.output_gradients) / quantity_of_samples
        
        #dLoss/dBias
        self.bias_gradients = np.sum(self.output_gradients, axis=0) / quantity_of_samples

        #dloss/dPrev_activations
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

    neural_network.train(x_train, y_train, epochs=10, learning_rate=0.01, batch_size=32, debug_one_backward_step=True)

    accuracy = neural_network.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")