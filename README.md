# simple-multilayer-perceptron

Multi-Layer Perceptron implementation in Typescript and Node using the Tensorflow library

# Introduction

Both the OOP class and functional equivalent are meant to be an abstraction of an Multi-Layer Perceptron inspired in the
'Machine Learning An Algorithmic Perspective' by Stephen Marsland in Typescript (Node)
using the Tensorflow library.

The book has its own implementation using Python and Numpy you can check it out here:
https://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html

This project is inspired by the following project https://github.com/bolt12/mlp-tf-node by Armando Santos.
My goals were to make especially the functional version of this simple perceptron to be more idiomatic typescript.

In this moment the MLP class is not very costumizable since you can only
set:

- the hidden layer activation function (sigmoid is the default)
- the output layer activation function (linear is the default)
- the learning rate (0.25)
- the training algorithm (sgd, momentum, adam)
- the loss function (meanSquaredError)
- number of hidden layers (1 is the default)
