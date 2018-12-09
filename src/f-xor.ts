import { tensor } from '@tensorflow/tfjs';

import {
  createModel,
  trainModel,
  getConfMatrixAndPrecision,
  getInputLayerShape,
  getOutputUnits
} from './f-mlp';

const inputTensor = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
const targetTensor = tensor([[0, 1, 1, 0]]);

const m = createModel({
  inputLayerShape: getInputLayerShape(inputTensor),
  outputUnits: getOutputUnits(targetTensor),
  hiddenUnits: 5,
  outputFunction: 'sigmoid',
  training: 'adam',
  numHiddenLayers: 4,
  learningRate: 0.1
});

trainModel({
  m,
  epochs: 1000,
  getInputForEpoch: _epochNum => inputTensor,
  getTargetForEpoch: _epochNum => targetTensor,
  validationSplit: 0
}).then(() => {
  const { confusionMatrix, precision } = getConfMatrixAndPrecision(
    m,
    inputTensor,
    targetTensor
  );
  confusionMatrix.print();
  console.log('Precision: ', precision * 100 + '%');
});
