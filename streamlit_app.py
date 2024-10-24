import streamlit as st
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_function):
        super(NeuralNetwork, self).__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        layers = []

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation_function)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def generate_data(func, x_values, y_values):
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_values = func(x_grid, y_grid)
    return x_grid, y_grid, z_values

def plot_math_function_and_approximation(func, x_values, y_values, model):
    x_grid, y_grid, z_values = generate_data(func, x_values, y_values)

    fig = go.Figure()

    # Plot the original mathematical function using a scatter plot
    fig.add_trace(go.Scatter3d(
        x=x_grid.flatten(),
        y=y_grid.flatten(),
        z=z_values.flatten(),
        mode='markers',
        marker=dict(size=2, opacity=1, color='blue'),
        name='Original Function'
    ))

    # Generate predictions using the neural network
    inputs = torch.tensor(np.column_stack((x_grid.flatten(), y_grid.flatten())), dtype=torch.float32)
    predictions = model(inputs).detach().numpy()
    predictions = predictions.reshape(x_grid.shape)

    fig.add_trace(go.Surface(z=predictions, x=x_values, y=y_values, name='Neural Network Approximation',opacity=0.8))

    fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='f(x,y)'),
                      margin=dict(l=0, r=0, b=0, t=40),
                      scene_camera=dict(eye=dict(x=1.87, y=0.88, z=-0.64)))

    return fig

def main():
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("NN Playground")
    st.write("Neural Network Playground is an interactive tool designed to help users understand the importance of tuning neural network hyperparameters. The app takes an analytical expression in the form f(x, y) = x² + y² and creates a neural network (NN) to approximate the surface of the function f(x, y). To achieve this, the function is evaluated at random values of x and y to generate a set of training points, which are then used to train the NN. Users can experiment with various NN hyperparameters, such as the number of layers, the number of neurons per layer, activation functions, learning rates, and more. The app provides real-time feedback on NN performance as these parameters are adjusted, offering valuable insights into how they affect the network's behavior. This tool is ideal for anyone interested in exploring and visualizing the impact of neural network design choices.")
    
    st.sidebar.header('Analytical Expression')
    
    # Get user input for the mathematical function
    expression = st.sidebar.text_input("Enter a mathematical expression in terms of x and y (e.g., x^2 + y^2):",value="(x-y)**2")

    # Create a lambda function from the user input
    try:
        func = lambda x, y: eval(expression)
    except Exception as e:
        st.sidebar.error(f"Error in the entered expression: {e}")
        return

    # Get user input for the range of values
    x_min, x_max = st.sidebar.slider("Select range for X-axis:", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Select range for Y-axis:", -10.0, 10.0, (-5.0, 5.0))
    
    num_of_points_in_space = st.sidebar.slider("Num of points in sample space:", 100, 10000, 100, 100)
    num_of_points_in_space = int(np.sqrt(num_of_points_in_space))
    # Generate values for the plot
    x_values = np.linspace(x_min, x_max, num_of_points_in_space)
    y_values = np.linspace(y_min, y_max, num_of_points_in_space)
    
    test_data_size = st.sidebar.slider("Test dataset size:", 0.2, 0.5, 0.1, 0.1)

    # Generate data for training the neural network
    x_grid, y_grid, z_values = generate_data(func, x_values, y_values)
    X_train, X_temp, y_train, y_temp = train_test_split(
        np.column_stack((x_grid.flatten(), y_grid.flatten())),
        z_values.flatten(),
        test_size=test_data_size,
        random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42
    )
    st.sidebar.header('Neural Network Paramters')
    # Get user input for neural network parameters
    learning_rate = st.sidebar.number_input("Enter learning rate:", min_value=0.001, max_value=0.9, value=0.01, step=0.01)
    hidden_layer_sizes = st.sidebar.text_input("Enter hidden layer sizes (comma-separated):", "8").split(',')
    hidden_layer_sizes = [int(size) for size in hidden_layer_sizes]
    activation_function_name = st.sidebar.selectbox("Select activation function:", ['ReLU', 'Sigmoid', 'Tanh'], index=0)
    activation_function = getattr(nn, activation_function_name)()
    loss_function_name = st.sidebar.selectbox("Select loss function:", ['MSELoss', 'L1Loss'], index=0)
    loss_function = getattr(nn, loss_function_name)()
    batch_size = st.sidebar.slider("Select batch size:", 1, 256, 32, 1)

    # Create and train the neural network
    input_size = X_train.shape[1]
    output_size = 1
    model = NeuralNetwork(input_size, hidden_layer_sizes, output_size, activation_function)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = st.sidebar.slider("Select number of training epochs:", 1, 100, 10)

    # Create a line chart for displaying the loss during training using matplotlib
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation and Training Loss')

    for epoch in range(epochs):
        for batch_start in range(0, len(X_train), batch_size):
            inputs = torch.tensor(X_train[batch_start:batch_start+batch_size], dtype=torch.float32)
            labels = torch.tensor(y_train[batch_start:batch_start+batch_size], dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        trainplot = plt.plot(epoch + 1, loss.item(), marker='.', color='r',label="train set")
        # Update the line chart with the latest validation loss
        validation_loss = loss_function(model(torch.tensor(X_val, dtype=torch.float32)),
                                        torch.tensor(y_val, dtype=torch.float32).view(-1, 1)).item()
        validplot = plt.plot(epoch + 1, validation_loss, marker='.', color='b',label="validation set")

    plt.legend(["train set","validation set"])

    # Plot the original mathematical function and neural network approximation
    fig = plot_math_function_and_approximation(func, x_values, y_values, model)
    st.plotly_chart(fig)

    # Display the matplotlib chart
    st.pyplot(plt)

if __name__ == "__main__":
    main()

