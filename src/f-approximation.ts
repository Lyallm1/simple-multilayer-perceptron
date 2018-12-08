import { tensor } from '@tensorflow/tfjs';

import {
  createModel,
  getInputLayerShape,
  getOutputUnits,
  earlyStoppingTraining,
  predict
} from './f-mlp';

const inputTensor = tensor([[0], [1], [2], [3], [4]]);
const targetTensor = tensor([[1, 2, 3, 4, 5]]);

const m = createModel({
  inputLayerShape: getInputLayerShape(inputTensor),
  outputUnits: getOutputUnits(targetTensor),
  hiddenUnits: 5,
  training: 'sgd'
});

earlyStoppingTraining({
  m,
  epochs: 2000,
  threshold: 0.000001,
  validationSplit: 0,
  getInputForEpoch: (_epochNum: number) => inputTensor,
  getTargetForEpoch: (_epochNum: number) => targetTensor
}).then(h => {
  predict({ m, input: [[0]] });
  predict({ m, input: [[1]] });
  predict({ m, input: [[2]] });
  predict({ m, input: [[3]] });
  predict({ m, input: [[4]] });
});
