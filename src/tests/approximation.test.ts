import { tensor } from '@tensorflow/tfjs';

import {
  createModel,
  getInputLayerShape,
  getOutputUnits,
  earlyStoppingTraining,
  predict
} from '../mlp';

const inputTensor = tensor([[0], [1], [2], [3], [4]]);
const targetTensor = tensor([[1, 2, 3, 4, 5]]);

const modelInstance = createModel({
  inputLayerShape: getInputLayerShape(inputTensor),
  outputUnits: getOutputUnits(targetTensor),
  hiddenUnits: 5,
  training: 'sgd'
});

test('early stopping prediction', () => {
  earlyStoppingTraining({
    modelInstance,
    trainingIterations: 2000,
    threshold: 0.000001,
    validationSplit: 0,
    getInputForTrainingIteration: (trainingIterationNum: number) =>
      inputTensor,
    getTargetForTrainingIteration: (trainingIterationNum: number) =>
      targetTensor
  }).then(h => {
    predict({ modelInstance, inputTensorLike: [[0]] });
    predict({ modelInstance, inputTensorLike: [[1]] });
    predict({ modelInstance, inputTensorLike: [[2]] });
    predict({ modelInstance, inputTensorLike: [[3]] });
    expect(
      Math.ceil(predict({ modelInstance, inputTensorLike: [[4]] })[0][0])
    ).toBe(5);
    console.log(predict({ modelInstance, inputTensorLike: [[4]] })[0][0]);
  });
});
