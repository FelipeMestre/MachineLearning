# Debugging My First Neural Network Training Loop (MNIST): A Postmortem

I got stuck implementing training for a neural network “from scratch” (NumPy). The forward pass looked plausible, but the backward pass, tensor shapes, and label/loss conventions kept breaking. This post lists the concrete mistakes I made and the exact fixes that got the model training.

---

### What I was building

- **Task**: MNIST classification (10 classes)
- **Input**: flattened images → shape `(batch_size, 784)`
- **Hidden layers**: `tanh`
- **Output**: `softmax` → shape `(batch_size, 10)`
- **Loss**: cross-entropy
- **Training loop**: forward → backward → update weights

---

### Error 1 — Wrong weight shapes (vectors instead of matrices)

- **Symptom**: errors like “can’t multiply `(32,784)` by `(100,)`”.
- **Cause**: weights were initialized as 1D vectors `(size,)` instead of a matrix.
- **Fix**:
  - `weights` must be `(n_in, n_out)`
  - `bias` must be `(n_out,)` (or `(1, n_out)`)

This means the **first** hidden layer uses `weights` shaped `(784, hidden_size)` and the next hidden layer uses `(hidden_size, hidden_size)`.

---

### Error 2 — Inconsistent input shape for single samples

- **Symptom**: single-sample forward/backward broke while batches worked.
- **Cause**: feeding a single example as `(784,)` instead of `(1,784)`.
- **Fix**: force inputs to always be 2D:
  - if `inputs.ndim == 1`: reshape to `(1, -1)`

---

### Error 3 — Forward pass didn’t actually chain through layers

- **Symptom**: network behaved like it wasn’t stacking layers.
- **Cause**: the loop computed each layer output but didn’t feed it into the next layer.
- **Fix**: update the activations each step:

  - correct pattern: `activations = hidden_layer.forward(activations)`

---

### Error 4 — Output layer skipped its own weights

- **Symptom**: training felt meaningless; caches were missing; gradients didn’t match the math.
- **Cause**: `OutputLayer.forward()` returned `softmax(inputs)` directly, skipping `Z = XW + b`.
- **Fix**: output layer must do:
  - cache `previous_layer_activations`
  - compute `layer_activations = X @ weights + bias`
  - compute `layer_outputs = softmax(layer_activations)`

---

### Error 5 — Using an incorrect “softmax derivative”

- **Symptom**: wrong learning signal / confusion about the gradient.
- **Cause**: using `softmax_derivative(x) = x*(1-x)` (that’s sigmoid-like, not softmax).
- **Fix**: with **softmax + cross-entropy**, use the standard simplification:
  - **one-hot targets**: `dZ = A - Y`
  - **sparse targets (indices)**: `dZ = A; dZ[i, y_i] -= 1`

This is why you “subtract 1 in the correct class”: it’s just implementing `A - Y` efficiently.

---

### Error 6 — Passing the loss into backward (instead of targets)

- **Symptom**: broadcast error like `(32,10) - (32,)`.
- **Cause**: computing a per-sample loss vector (shape `(batch,)`) and calling `backward(error)`.
- **Fix**: backward needs **targets**, not a loss vector:

- compute loss for logging
- call `backward(batch_targets_batch)`
- then update weights

---

### Error 7 — Mixing sparse and one-hot label conventions

- **Symptom**: code expected one thing but data provided another.
- **Cause**: using a “sparse” cross-entropy function while training with one-hot labels.
- **Fix**: pick one convention and keep it consistent:
  - **one-hot labels** → categorical cross-entropy + `dZ = A - Y`
  - **index labels** → sparse cross-entropy + `dZ[i, y_i] -= 1`

---

### Error 8 — Backward loop variable mistakes (gradient not flowing)

- **Symptom**: runtime errors or gradients not propagating properly.
- **Cause**: using an undefined variable (e.g., `activation_gradients`) or returning the wrong gradient.
- **Fix**: backward must:
  1. start from the output layer gradient
  2. pass the returned gradient into each hidden layer in reverse order

---

### Error 9 — Gradient arrays with wrong shapes

- **Symptom**: weight updates didn’t match parameter shapes.
- **Cause**: initializing `weights_gradients` as `(size,)` instead of `(n_in, n_out)`.
- **Fix**:
  - `weights_gradients = zeros((previous_layer_size, size))`
  - `bias_gradients = zeros((size,))`

---

### Error 10 — Accuracy calculation didn’t match one-hot targets

- **Symptom**: evaluation/accuracy logic was incorrect.
- **Cause**: comparing predictions against one-hot targets without `argmax`.
- **Fix**:
  - `argmax(predictions, axis=1) == argmax(targets, axis=1)`

---

### Error 11 — TensorFlow imports made the module un-importable

- **Symptom**: `ModuleNotFoundError: No module named 'tensorflow'` when importing the file.
- **Cause**: unconditional `import tensorflow as tf` at module import time, plus running MNIST code on import.
- **Fix**:
  - wrap TF imports in `try/except`
  - move training/demo code under `if __name__ == "__main__":`

This makes the neural network classes usable even without TensorFlow installed.

---

### The final mental model (what backward must do)

For each layer in reverse order:

- **Compute local gradient** (e.g., `dZ = dA * tanh'(Z)` for tanh)
- **Compute parameter gradients**:
  - `dW = A_prev.T @ dZ / m`
  - `db = sum(dZ) / m`
- **Propagate gradient backward**:
  - `dA_prev = dZ @ W.T`

And then, **after backward**, apply:

- `W -= lr * dW`
- `b -= lr * db`

---

### Takeaway

Most issues weren’t “deep math” problems—they were **shape discipline** and **API discipline** problems:

- always use 2D batches
- keep label format consistent (one-hot vs indices)
- never pass a loss vector into backward
- output layer must be a real dense layer (`XW + b`) before softmax