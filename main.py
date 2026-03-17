import numpy as np
from Perceptron import Perceptron


if __name__ == '__main__':
    # Input data for training the Perceptron
    inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

    # Different targets, depending on which propositional logic connective the Perceptron
    # is supposed to learn.
    targets_or = np.array([1, 1, 1, 0])
    targets_and = np.array([0, 0, 1, 0])
    targets_nand = np.array([1, 1, 0, 1])
    targets_nor = np.array([0, 0, 0, 1])


    #initiating different Perceptrons, one for each connective
    perceptron_or = Perceptron(np.zeros(2), 0, 1)
    perceptron_and = Perceptron(np.zeros(2), 0, 1)
    perceptron_nand = Perceptron(np.zeros(2), 0, 1)
    perceptron_nor = Perceptron(np.zeros(2), 0, 1)


    print(f"---Training OR ---")
    perceptron_or.train_n_epochs(4, inputs, targets_or, True)

    print(f"---Training AND ---")
    perceptron_and.train_n_epochs(7, inputs, targets_and, True)

    print(f"---Training NAND ---")
    perceptron_nand.train_n_epochs(8, inputs, targets_nand, True)

    print(f"---Training NOR ---")
    perceptron_nor.train_n_epochs(5, inputs, targets_nor, True)














