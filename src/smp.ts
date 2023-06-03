import '@tensorflow/tfjs-node';

import * as tf from '@tensorflow/tfjs';

import { TensorLike } from '@tensorflow/tfjs-core/dist/types';

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
export interface TrainSmpOptions {
  smp: tf.Model;
  trainingIterations: number;
  validationSplit?: number;
  epochs?: number;
  getInputForTrainingIteration: (trainingIterationNum: number) => tf.Tensor;
  getTargetForTrainingIteration: (trainingIterationNum: number) => tf.Tensor;
}

export const getInputLayerShape = (inputTensor: tf.Tensor) => [inputTensor.shape[1]], getOutputUnits = (targetTensor: tf.Tensor) => targetTensor.shape[0],
createSmp = ({ inputLayerShape, outputUnits, hiddenUnits, training = 'adam', hiddenFunction = 'sigmoid', outputFunction = 'linear', lossFunction = 'meanSquaredError', learningRate = 0.25, numHiddenLayers = 1 }: CreateSmpOptions) => {
  const inputLayer = tf.input({ shape: inputLayerShape }), hiddenLayers = [];
  for (let i = 0; i < numHiddenLayers; i++) hiddenLayers[i] = tf.layers.dense({ units: hiddenUnits, activation: hiddenFunction, useBias: true }).apply(i === 0 ? inputLayer : hiddenLayers[i - 1]);
  const smp = tf.model({ inputs: inputLayer, outputs: tf.layers.dense({ units: outputUnits, activation: outputFunction, useBias: true }).apply(hiddenLayers[numHiddenLayers - 1]) as tf.SymbolicTensor });
  smp.compile({ optimizer: training === 'momentum' ? tf.train.momentum(learningRate, 0.9) : tf.train[training](learningRate), loss: lossFunction, metrics: ['accuracy'] });
  return smp;
}, trainSmp = async ({ smp: modelInstance, trainingIterations, validationSplit = 0.1, epochs = 1, getInputForTrainingIteration, getTargetForTrainingIteration }: TrainSmpOptions) => {
  let newValError = 1000000, history = null;
  for (let i = 0; i < trainingIterations; i++) {
    try {
      history = await modelInstance.fit(getInputForTrainingIteration(i), getTargetForTrainingIteration(i).transpose(), { epochs, validationSplit, shuffle: true });
    } catch (e) {
      console.log('error: ', e);
    }
    newValError = history.history.loss as number;
    if (i % 100 === 0) console.log('trainingIteration: ' + i + '\nloss: ' + newValError);
  }
  console.log('Training stopped ', newValError);
  return history;
}, getSmpConfMatrixAndPrecision = (smp: tf.Model, inputTensor: tf.Tensor, targetTensor: tf.Tensor) => {
  const outputs = tf.round(smp.predict(inputTensor) as tf.Tensor);
  let nClasses = targetTensor.buffer().get(targetTensor.argMax(1).dataSync()[0]) as number;
  if (nClasses === 1) nClasses = 2;
  const cm = tf.zeros([nClasses, nClasses]).buffer();
  for (let i = 0; i < nClasses; i++) for (let j = 0; j < nClasses; j++) cm.set(targetTensor.toBool().equal(tf.fill(targetTensor.shape, j).toBool()).toFloat().matMul(outputs.toBool().equal(tf.fill([outputs.shape[0], 1], i).toBool()).toFloat()).sum().dataSync() as unknown as number, i, j);
  return { confusionMatrix: cm.toTensor(), precision: tf.oneHot(tf.linspace(0, cm.shape[0] - 1, cm.shape[0]).toInt(), cm.shape[0]).toFloat().mul(cm.toTensor().toFloat()).sum().toFloat().div(cm.toTensor().sum()).dataSync() as unknown as number };
}, smpPredict = ({ smp, inputTensorLike }: { smp: tf.Model, inputTensorLike: TensorLike }) => {
  const logits = Array((smp.predict(tf.tensor(inputTensorLike)) as tf.Tensor).dataSync());
  console.log('Prediction: ', logits);
  return logits;
}, smpEarlyStoppingTraining = async ({ smp: modelInstance, trainingIterations, threshold, validationSplit = 0.1, getInputForTrainingIteration, getTargetForTrainingIteration }: {
  smp: tf.Model, trainingIterations: number, threshold: number, validationSplit: number, getInputForTrainingIteration: (trainingIterationNum: number) => tf.Tensor, getTargetForTrainingIteration: (trainingIterationNum: number) => tf.Tensor
}) => {
  let oldValError1 = 100002, oldValError2 = 100001, newValError = 100000, count = 0, history = null;
  while ((count < trainingIterations && oldValError1 - newValError > threshold) || oldValError2 - oldValError1 > threshold) {
    count++;
    history = await modelInstance.fit(getInputForTrainingIteration(count), getTargetForTrainingIteration(count).transpose(), { validationSplit, shuffle: true });
    [oldValError2, oldValError1, newValError] = [oldValError1, newValError, history.history.loss as number];
    if (count % 100 === 0) console.log('training iteration: ' + count + '\nloss: ' + newValError);
  }
  console.log('Training stopped ', newValError, oldValError1, oldValError2, count);
  return history;
};
