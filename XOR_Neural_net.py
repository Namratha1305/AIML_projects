import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # type: ignore

# --- 1. Define the XOR data ---
# Input data (X) for the XOR problem
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target output (y) for the XOR problem
y = np.array([[0],
              [1],
              [1],
              [0]])

# --- 2. Define the Neural Network Architecture and Functions ---

class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        # Initialize weights and biases with small random values
        self.weights_input_hidden = np.random.randn(input_neurons, hidden_neurons)
        self.bias_hidden = np.zeros((1, hidden_neurons))
        self.weights_hidden_output = np.random.randn(hidden_neurons, output_neurons)
        self.bias_output = np.zeros((1, output_neurons))

    # Define the Sigmoid Activation Function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Define the Derivative of the Sigmoid Function (for Backpropagation)
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # --- 3. Implement the Training Process (Forward Propagation and Backpropagation) ---
    def train(self, X, y, epochs, learning_rate):
        loss_history = []
        for epoch in range(epochs):
            # --- 4. Forward Propagation ---
            # Input to Hidden Layer
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            # Hidden to Output Layer
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            predicted_output = self.sigmoid(output_layer_input)

            # --- 5. Backpropagation (The Core Learning Part) ---
            # Calculate Output Layer Error and Delta
            output_error = y - predicted_output
            output_delta = output_error * self.sigmoid_derivative(predicted_output)

            # Calculate Hidden Layer Error and Delta
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_output)

            # Update Weights and Biases
            self.weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
            self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
            self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

            # --- 6. Training Loop: Print the loss ---
            if epoch % 1000 == 0:
                loss = np.mean(np.square(output_error))
                loss_history.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return loss_history

    # --- 7. Testing: Make predictions ---
    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        predicted_output = self.sigmoid(output_layer_input)
        return predicted_output

# --- 8. Main Execution ---

# Set hyperparameters
epochs = 20000
learning_rate = 0.1
input_neurons = 2
hidden_neurons = 3  # You can experiment with 2 or 3 neurons
output_neurons = 1

# Create an instance of the Neural Network
nn = NeuralNetwork(input_neurons, hidden_neurons, output_neurons)

# Train the network
print("Starting training...")
loss_history = nn.train(X, y, epochs, learning_rate)
print("Training finished.")

# Make predictions on the XOR data
predictions = nn.predict(X)

# Print the final results
print("\n--- Final Predictions ---")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {predictions[i][0]:.4f}, Target: {y[i][0]}")

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title('Training Loss Over Time')
plt.xlabel('Epochs (in thousands)')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()