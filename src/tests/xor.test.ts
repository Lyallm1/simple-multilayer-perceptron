import { tensor } from '@tensorflow/tfjs';

import {
  createSmp,
  trainSmp,
  getSmpConfMatrixAndPrecision,
  getInputLayerShape,
  getOutputUnits
} from '../smp';

const inputTensor = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
const targetTensor = tensor([[0, 1, 1, 0]]);

const smp = createSmp({
  inputLayerShape: getInputLayerShape(inputTensor),
  outputUnits: getOutputUnits(targetTensor),
  hiddenUnits: 5,
  outputFunction: 'sigmoid',
  training: 'adam',
  numHiddenLayers: 4,
  learningRate: 0.1
});

test('smp xor', () => {
  trainSmp({
    smp,
    trainingIterations: 1000,
    epochs: 2,
    getInputForTrainingIteration: (trainingIterationNum: number) =>
      inputTensor,
    getTargetForTrainingIteration: (trainingIterationNum: number) =>
      targetTensor,
    validationSplit: 0
  }).then(() => {
    const { confusionMatrix, precision } = getSmpConfMatrixAndPrecision(
      smp,
      inputTensor,
      targetTensor
    );
    confusionMatrix.print();
    expect(precision * 100).toBe(100);

    console.log('Precision: ', precision * 100 + '%');
  });
});
