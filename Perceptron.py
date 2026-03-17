
import numpy as np

class Perceptron:
    def __init__(self, weights: np.ndarray, bias: float, learning_rate: float):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate



    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias




    def compute_activation(self, inputs: np.ndarray,) -> float:
        """
           Computes the activation of the Perceptron
           The Activation is the sum of the weighted inputs plus bias
           Args:   inputs (np.array): Numpy array containing the inputs to the perceptron

           Returns: float: float containing the Perceptron's activation
        """
        return np.dot(inputs,self.weights)+self.bias


    def activation_function(self, activation: float)-> int:
        """
            Apply the activation function to the activation value
            This case uses the Heaviside Function as the activation function
            Args: activation (float): The activation value

            Returns: int: 1 or 0
        """
        return int(np.heaviside(activation, 0))


    def output(self, inputs: np.ndarray):
        """
           Computes the Perceptrons output
           Args:
                  inputs (np.ndarray): Numpy array containing the inputs to the perceptron

           Returns: int: 1  or 0
        """
        activation = self.compute_activation(inputs)
        return self.activation_function(activation)




    def update_weights(self, inputs: np.ndarray, target: float, output: float):
        """
           Updates the Perceptrons Weights by computing new Weights based on the
           Output and the actual target
           Args:
                  inputs (np.ndarray): Numpy array containing the inputs to the perceptron
                  target (float): Float containing the target value
                  output (float): Float containing the Perceptrons output

           Returns: self.weights (np.ndarray): Weights of the Perceptron after being updated
        """
        deltaW = (target-output)*inputs*self.learning_rate
        self.weights = deltaW + self.weights
        return self.weights


    def update_bias(self, target: float, output: float)->float:
        """
            Updates the Perceptrons Bias by computing the new Bias based on the
            output and the actual target
            Args: target (float): Float containing the target value
                  output (float): Float containing the Perceptrons output

            Returns: self.bias (float): Bias of the Perceptron after being updated
        """
        self.bias = self.bias + (target-output)*self.learning_rate
        return self.bias


    def do_one_trial(self, inputs: np.ndarray, target: float):
        """
            Runs one trial with the given inputs / target
            Args:   inputs (np.ndarray): Numpy array containing the inputs to the perceptron
                    target (float): Float containing the target value

            Returns:    trainedWeights (np.ndarray): updated weights after one trial
                        trained Bias (float): updated bias after one trial
                        output int: 1 or 0, output of the perceptron
        """
        output = self.output(inputs)
        trainedWeights = self.update_weights(inputs, target, output)
        trainedBias = self.update_bias(target, output)
        return trainedWeights, trainedBias, output



    @staticmethod
    def error_rate(targets: np.ndarray, outputs: list) -> float:
        """
            Calculate the error rate of the Perceptron
            Args:   targets (np.ndarray): Numpy Array containing the targets for a given input
                    outputs (list): Perceptrons outputs for given inputs

            Return: error_rate (float): error rate of the perceptron
        """
        outputs_array = np.array(outputs)
        sum_correct = np.sum(targets == outputs_array)
        accuracy = sum_correct / len(targets)
        error_rate = 1 - accuracy
        return error_rate



    def train_one_epoch(self, inputs: np.ndarray, targets: np.ndarray, verbose: bool):
        """
            Trains the Perceptron on all given inputs once
            Args:   inputs (np.ndarray): Numpy array containing the inputs to the perceptron
                    targets (np.ndarray): Numpy Array containing the targets for a given input
                    verbose (bool): if true, methods prints out weights, bias and error rate
                                    after one epoch

            Returns:    total_outputs (list):   every output the Perceptron calculated for a
                                                given input
        """
        total_outputs = []

        for current_input,target in zip(inputs, targets):
            self.weights, self.bias, output = self.do_one_trial(current_input, target)
            total_outputs.append(output)

        errors = self.error_rate(targets, total_outputs)

        if verbose:
            print(f"Weights after one Epoch: {self.weights}"
                  f" Bias after one Epoch: {self.bias}"
                  f" Error Rate: {errors}")

        return total_outputs



    def train_n_epochs(self, epochs: int, inputs: np.ndarray, targets: np.ndarray, verbose: bool):
        """
            Trains the Perceptron multiple times on given inputs
            Args:   epochs (int): amount of epochs the perceptron is to be trained on
                    inputs (np.ndarray): Numpy array containing the inputs to the perceptron
                    targets (np.ndarray): Numpy Array containing the targets for a given input
                    verbose (bool): if true, methods prints out weights, bias and error rate
                                    after one epoch
        """
        for i in range(epochs):
            self.train_one_epoch(inputs, targets, verbose)