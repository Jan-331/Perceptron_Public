# Perceptron

Simple implementation of a Perceptron in Python with an object-oriented approach from scratch, 
using only numpy. The Perceptron implemented here is able to learn propositional logic connectives
(OR, AND, NOR, NAND). 
To showcase the improvement of the Perceptrons performance over time I implemented a function which 
computes the error rate after each iteration through the inputs.

## How it works:

The Perceptron takes a set of inputs $x$, multiplies them by corresponding weights $w$, and adds a bias $b$.
The sum is passed through a Heaviside step function:
- If $w \cdot x + b \ge 0$, output is 1
- If $w \cdot x + b < 0$, output is 0

Weights and bias are updated iteratively based on the difference between the predicted output and the actual target.

## Usage

Only prerequisites are Numpy and Python.
To run the project simply execute the Python script. The script initializes four different perceptrons and 
trains them on the truth tables of OR, AND, NAND, and NOR gates.

The console will then output the learned weights, bias and error rate after each epoch, demonstrating
how the Perceptron minimizes the  error rate to 0.0

## Limitations

As a Single-Layer Perceptron, this model can only solve linearly separable problems.
It successfully learns AND, OR, NAND, and NOR gates. However, 
it cannot learn the XOR (Exclusive OR) gate, as its outputs cannot be separated by a single line.
Solving XOR requires a Multi-Layer Perceptron (MLP).