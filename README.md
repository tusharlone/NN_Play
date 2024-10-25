# Neural Network Playground
Neural Network Playground is an interactive tool designed to help users understand the importance of tuning neural network hyperparameters. The app takes an analytical expression in the form f(x, y) = x² + y² (continuous) and creates a neural network (NN) to approximate a smooth surface to the function f(x, y). To achieve this, the function is evaluated for specific values of x and y sampled using the grid sampling method to generate a set of training points, which are then used to train the NN. Users can experiment with various NN hyperparameters, such as the number of layers, the number of neurons per layer, activation functions, learning rates, and more. The app provides real-time feedback on NN performance as these parameters are adjusted, offering valuable insights into how they affect the network's behavior. This tool is ideal for anyone interested in exploring and visualizing the impact of neural network design choices.

This app approximates a smooth surface to any analytical expression you provide as input. In the sidebar menu, you can enter an expression in the form of x² + y², and the app will approximate the surface for the function f = x² + y². You can customize the ranges for variables x and y and specify the number of points to sample for training the neural network (NN). Additionally, you can adjust the NN hyperparameters in the sidebar, experimenting with different values to see how the trained NN improves in approximating the given analytical expression. In the 3D interactable graph, the training points are shown as blue dots, representing the data used to guide the NN's learning, and the surface represents the NN approximation.
