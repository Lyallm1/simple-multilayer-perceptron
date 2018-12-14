import { tensor } from '@tensorflow/tfjs';

import {
  createModel,
  trainModel,
  getConfMatrixAndPrecision,
  getInputLayerShape,
  getOutputUnits
} from '../mlp';

const inputTensor = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
const targetTensor = tensor([[0, 1, 1, 0]]);

const modelInstance = createModel({
  inputLayerShape: getInputLayerShape(inputTensor),
  outputUnits: getOutputUnits(targetTensor),
  hiddenUnits: 5,
  outputFunction: 'sigmoid',
  training: 'adam',
  numHiddenLayers: 4,
  learningRate: 0.1
});

test('precision', () => {
  trainModel({
    modelInstance,
    trainingIterations: 1000,
    epochs: 2,
    getInputForTrainingIteration: (trainingIterationNum: number) =>
      inputTensor,
    getTargetForTrainingIteration: (trainingIterationNum: number) =>
      targetTensor,
    validationSplit: 0
  }).then(() => {
    const { confusionMatrix, precision } = getConfMatrixAndPrecision(
      modelInstance,
      inputTensor,
      targetTensor
    );
    confusionMatrix.print();
    expect(precision * 100).toBe(100);

    console.log('Precision: ', precision * 100 + '%');
  });
});
