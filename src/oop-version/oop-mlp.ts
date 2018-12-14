import * as tf from '@tensorflow/tfjs';
import { TensorLike } from '@tensorflow/tfjs-core/dist/types';
import '@tensorflow/tfjs-node';

/*
This class is meant to be an abstraction of an Multi-Layer Perceptron inspired in the 
'Machine Learning An Algorithimic Perspective' by Stephen Marsland in Javascript (Node JS) 
using the Tensorflow library.

The book has its own implementation using Python and Numpy you can check it out here:
https://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html

The code below is an adaptation of https://github.com/bolt12/mlp-tf-node by Armando Santos

In this moment the MLP class is not very costumizable since you can only
set:
- the hidden layer activation function (sigmoid is the default)
- the output layer activation function (linear is the default)
- the learning rate ()
- the training algorithm (sgd, momentum, adam)
- the loss function
*/

export class MLPerceptron {
  public hiddenFunction: any;
  public outputFunction: any;
  public learningRate: any;
  public lossFunction: any;
  public inputs: any;
  public targets: any;
  public inputLayer: any;
  public hiddenLayer: any;
  public outputLayer: any;
  public training: any;
  public model: any;
  public precision: any;

  constructor(
    inputs: any,
    targets: any,
    hidden: any,
    training: any,
    hfunction: any,
    outfunction: any,
    lossfunction: any,
    learningRate: any
  ) {
    this.hiddenFunction =
      typeof hfunction !== 'undefined' ? hfunction : 'sigmoid';
    this.outputFunction =
      typeof outfunction !== 'undefined' ? outfunction : 'linear';
    this.learningRate =
      typeof learningRate !== 'undefined' ? learningRate : 0.25;
    this.lossFunction =
      typeof lossfunction !== 'undefined'
        ? lossfunction
        : 'meanSquaredError';
    this.inputs = tf.tensor(inputs);
    this.targets = tf.tensor(targets);
    this.inputLayer = tf.input({ shape: [this.inputs.shape[1]] });
    this.hiddenLayer = tf.layers
      .dense({
        units: hidden,
        activation: this.hiddenFunction,
        useBias: true
      })
      .apply(this.inputLayer);
    this.outputLayer = tf.layers
      .dense({
        units: this.targets.shape[0],
        activation: this.outputFunction,
        useBias: true
      })
      .apply(this.hiddenLayer);

    if (training === 'sgd') {
      this.training = tf.train.sgd(this.learningRate);
    } else if (training === 'momentum') {
      this.training = tf.train.momentum(this.learningRate, 0.9);
    } else {
      this.training = tf.train.adam(this.learningRate);
    }
    this.model = tf.model({
      inputs: this.inputLayer,
      outputs: this.outputLayer
    });

    /* TODO: Let the loss function to be parameterized */
    this.model.compile({
      optimizer: this.training,
      loss: this.lossFunction,
      metrics: ['accuracy']
    });
  }

  /* 
    Normal MLP training
    Receives the number of iterations to train.
    Uses a validationSplit of 0.25. TODO: parameterize the validationSplit
  */
  public async train(trainingIterations: number, validationSplit: number) {
    validationSplit =
      typeof validationSplit !== 'undefined' ? validationSplit : 0.1;
    console.log('v spli: ', validationSplit);
    let newValError = 1000000;
    let history = null;
    for (let i = 0; i < trainingIterations; i++) {
      history = await this.model.fit(
        this.inputs,
        this.targets.transpose(),
        { validationSplit, shuffle: true }
      );
      newValError = history.history.loss;
      if (i % 100 === 0) {
        console.log(
          'training iterations: ' + i + '\nloss: ' + newValError
        );
      }
    }
    console.log('Training stopped ', newValError);
    return history;
  }

  /*
    Early Stopping training technique
    Receives the maximum training iterations and error threshold.
    This function will train the MLP until the loss value in the
    validation set is less than the treshold which means the network
    stopped learning about the inputs and start learning about the noise
    in the inputs.
  */
  public async earlyStoppingTraining(
    trainingIterations: number,
    threshold: number,
    validationSplit: number
  ) {
    validationSplit =
      typeof validationSplit !== 'undefined' ? validationSplit : 0.1;
    let oldValError1 = 100002;
    let oldValError2 = 100001;
    let newValError = 100000;

    let count = 0;
    let history = null;

    while (
      (count < trainingIterations &&
        oldValError1 - newValError > threshold) ||
      oldValError2 - oldValError1 > threshold
    ) {
      count += 1;
      history = await this.model.fit(
        this.inputs,
        this.targets.transpose(),
        { validationSplit, shuffle: true }
      );
      oldValError2 = oldValError1;
      oldValError1 = newValError;
      newValError = history.history.loss;
      if (count % 100 === 0) {
        console.log(
          'training iteration: ' + count + '\nloss: ' + newValError
        );
      }
    }
    console.log(
      'Training stopped ',
      newValError,
      oldValError1,
      oldValError2,
      count
    );
    return history;
  }

  /*
    Feeds forward the inputs.
  */
  public predict(inputTensorLike: TensorLike) {
    const input = tf.tensor(inputTensorLike);
    const predictOut = this.model.predict(input);
    const logits = Array(predictOut.dataSync());
    console.log('Prediction: ', logits);

    // Clean up
    input.dispose();
    return logits;
  }

  /*
    Calculates the confusion matrix for a
    set of inputs and targets.
    Adds the calculated precision as a class
    atribute.

    NOTE: Arguments must be instance of Tensor
  */
  public confMatrix(inputs: any, targets: any) {
    let outputs = this.model.predict(inputs);
    const indice = targets.argMax(1).dataSync();
    let nClasses = targets.buffer().get(indice[0]);

    if (nClasses === 1) {
      nClasses = 2;
    }
    outputs = tf.round(outputs);

    const cm = tf.zeros([nClasses, nClasses]).buffer();
    for (let i = 0; i < nClasses; i++) {
      for (let j = 0; j < nClasses; j++) {
        const mI = tf.fill([outputs.shape[0], 1], i);
        const mJ = tf.fill(targets.shape, j);
        const a = outputs
          .toBool()
          .equal(mI.toBool())
          .toFloat();
        const b = targets
          .toBool()
          .equal(mJ.toBool())
          .toFloat();
        const sum = b
          .matMul(a)
          .sum()
          .dataSync();

        // Clean up
        mI.dispose();
        mJ.dispose();
        a.dispose();
        b.dispose();

        cm.set(sum, i, j);
      }
    }

    // Calculate precision
    const trace = tf
      .oneHot(
        tf.linspace(0, cm.shape[0] - 1, cm.shape[0]).toInt(),
        cm.shape[0]
      )
      .toFloat()
      .mul(cm.toTensor().toFloat())
      .sum();
    const total = cm.toTensor().sum();
    const precision = tf.div(trace.toFloat(), total);
    this.precision = precision.dataSync();

    // Clean up
    precision.dispose();

    return cm.toTensor();
  }
}
