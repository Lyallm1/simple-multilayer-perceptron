import {
  tensor,
  Tensor,
  SymbolicTensor,
  Rank,
  model,
  Model,
  input,
  layers,
  train,
  round,
  zeros,
  fill,
  linspace,
  div,
  oneHot
} from '@tensorflow/tfjs';
import { TensorLike } from '@tensorflow/tfjs-core/dist/types';
import '@tensorflow/tfjs-node';

/*
This functional programming style set of functions are meant to be an abstraction
of an Multi-Layer Perceptron inspired in the 
'Machine Learning An Algorithimic Perspective' by Stephen Marsland in Javascript (Node JS) 
using the Tensorflow library.

The book has its own implementation using Python and Numpy you can check it out here:
https://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html

The code below is an adaptation of https://github.com/bolt12/mlp-tf-node by Armando Santos

In this moment the functions are not very costumizable since you can only
set:
- the hidden layer activation function (sigmoid is the default)
- the output layer activation function (linear is the default)
- the learning rate ()
- the training algorithm (sgd, momentum, adam)
- the loss function
*/

export const getInputLayerShape = (inputTensor: Tensor) => [
  inputTensor.shape[1]
];
export const getOutputUnits = (targetTensor: Tensor) =>
  targetTensor.shape[0];

export interface CreateSmpOptions {
  inputLayerShape: number[];
  outputUnits: number;
  hiddenUnits: number;
  training?: 'sgd' | 'momentum' | 'adam';
  hiddenFunction?: string;
  outputFunction?: string;
  lossFunction?: string;
  learningRate?: number;
  numHiddenLayers?: number;
}

export const createSmp = ({
  inputLayerShape,
  outputUnits,
  hiddenUnits,
  training = 'adam',
  hiddenFunction = 'sigmoid',
  outputFunction = 'linear',
  lossFunction = 'meanSquaredError',
  learningRate = 0.25,
  numHiddenLayers = 1
}: CreateSmpOptions) => {
  const inputLayer = input({
    shape: inputLayerShape
  });
  const hiddenLayers = [];

  for (let i = 0; i < numHiddenLayers; i++) {
    const applyLayer: any = i === 0 ? inputLayer : hiddenLayers[i - 1];
    hiddenLayers[i] = layers
      .dense({
        units: hiddenUnits,
        activation: hiddenFunction,
        useBias: true
      })
      .apply(applyLayer);
  }
  const outputLayer = layers
    .dense({
      units: outputUnits,
      activation: outputFunction,
      useBias: true
    })
    .apply(hiddenLayers[numHiddenLayers - 1]);
  const optimizer =
    training === 'momentum'
      ? train.momentum(learningRate, 0.9)
      : train[training](learningRate);
  const smp = model({
    inputs: inputLayer,
    outputs: outputLayer as SymbolicTensor
  });
  smp.compile({
    optimizer,
    loss: lossFunction,
    metrics: ['accuracy']
  });
  return smp;
};

export interface TrainSmpOptions {
  smp: Model;
  trainingIterations: number;
  validationSplit?: number;
  epochs?: number;
  getInputForTrainingIteration: (trainingIterationNum: number) => Tensor;
  getTargetForTrainingIteration: (trainingIterationNum: number) => Tensor;
}

export const trainSmp = async ({
  smp: modelInstance,
  trainingIterations,
  validationSplit = 0.1,
  epochs = 1,
  getInputForTrainingIteration,
  getTargetForTrainingIteration
}: TrainSmpOptions) => {
  let newValError = 1000000;
  let history = null;
  for (let i = 0; i < trainingIterations; i++) {
    try {
      history = await modelInstance.fit(
        getInputForTrainingIteration(i),
        getTargetForTrainingIteration(i).transpose(),
        {
          epochs,
          validationSplit,
          shuffle: true
        }
      );
    } catch (e) {
      console.log('error: ', e);
    }
    newValError = (history.history.loss as unknown) as number;
    if (i % 100 === 0) {
      console.log('trainingIteration: ' + i + '\nloss: ' + newValError);
    }
  }
  console.log('Training stopped ', newValError);
  return history;
};

export const getSmpConfMatrixAndPrecision = (
  smp: Model,
  inputTensor: Tensor,
  targetTensor: Tensor
) => {
  const outputs = round(smp.predict(inputTensor) as Tensor);
  const index = targetTensor.argMax(1).dataSync();
  let nClasses = targetTensor.buffer().get(index[0]) as number;

  if (nClasses === 1) {
    nClasses = 2;
  }

  const cm = zeros([nClasses, nClasses]).buffer();
  for (let i = 0; i < nClasses; i++) {
    for (let j = 0; j < nClasses; j++) {
      const mI = fill([outputs.shape[0], 1], i);
      const mJ = fill(targetTensor.shape, j);
      const a = outputs
        .toBool()
        .equal(mI.toBool())
        .toFloat();
      const b = targetTensor
        .toBool()
        .equal(mJ.toBool())
        .toFloat();
      const sum = (b
        .matMul(a as Tensor<Rank.R2>)
        .sum()
        .dataSync() as unknown) as number;

      // Clean up
      mI.dispose();
      mJ.dispose();
      a.dispose();
      b.dispose();

      cm.set(sum, i, j);
    }
  }

  // Calculate precision
  const trace = oneHot(
    linspace(0, cm.shape[0] - 1, cm.shape[0]).toInt(),
    cm.shape[0]
  )
    .toFloat()
    .mul(cm.toTensor().toFloat())
    .sum();
  const total = cm.toTensor().sum();
  const precisionTensor = div(trace.toFloat(), total);
  const precision = (precisionTensor.dataSync() as unknown) as number;

  // Clean up
  precisionTensor.dispose();

  return { confusionMatrix: cm.toTensor(), precision };
};

export const smpPredict = ({
  smp: smp,
  inputTensorLike
}: {
  smp: Model;
  inputTensorLike: TensorLike;
}) => {
  const inputTensor = tensor(inputTensorLike);
  const predictOut: any = smp.predict(inputTensor);
  const logits = Array(predictOut.dataSync());
  console.log('Prediction: ', logits);

  // Clean up
  inputTensor.dispose();
  return logits;
};

/*
    Early Stopping training technique
    Receives the maximum training iterations and error threshold.
    This function will train the MLP until the loss value in the
    validation set is less than the treshold which means the network
    stopped learning about the inputs and start learning about the noise
    in the inputs.
  */

interface SmpEarlyStoppingTrainingOptions {
  smp: Model;
  trainingIterations: number;
  threshold: number;
  validationSplit: number;
  getInputForTrainingIteration: (trainingIterationNum: number) => Tensor;
  getTargetForTrainingIteration: (trainingIterationNum: number) => Tensor;
}
export const smpEarlyStoppingTraining = async ({
  smp: modelInstance,
  trainingIterations,
  threshold,
  validationSplit = 0.1,
  getInputForTrainingIteration,
  getTargetForTrainingIteration
}: SmpEarlyStoppingTrainingOptions) => {
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
    count++;
    const inputTensor = getInputForTrainingIteration(count);
    const targetTensor = getTargetForTrainingIteration(count);
    history = await modelInstance.fit(
      inputTensor,
      targetTensor.transpose(),
      {
        validationSplit,
        shuffle: true
      }
    );
    oldValError2 = oldValError1;
    oldValError1 = newValError;
    newValError = (history.history.loss as unknown) as number;
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
};
