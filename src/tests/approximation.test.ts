import { tensor } from '@tensorflow/tfjs';

import {
  createSmp,
  getInputLayerShape,
  getOutputUnits,
  smpEarlyStoppingTraining,
  smpPredict
} from '../smp';

const inputTensor = tensor([[0], [1], [2], [3], [4]]);
const targetTensor = tensor([[1, 2, 3, 4, 5]]);

console.log(getOutputUnits(targetTensor));

const smp = createSmp({
  inputLayerShape: getInputLayerShape(inputTensor),
  outputUnits: getOutputUnits(targetTensor),
  hiddenUnits: 5,
  training: 'sgd'
});

test('smp early stopping prediction', () => {
  smpEarlyStoppingTraining({
    smp,
    trainingIterations: 2000,
    threshold: 0.000001,
    validationSplit: 0,
    getInputForTrainingIteration: (trainingIterationNum: number) =>
      inputTensor,
    getTargetForTrainingIteration: (trainingIterationNum: number) =>
      targetTensor
  }).then(h => {
    smpPredict({ smp, inputTensorLike: [[0]] });
    smpPredict({ smp, inputTensorLike: [[1]] });
    smpPredict({ smp, inputTensorLike: [[2]] });
    smpPredict({ smp, inputTensorLike: [[3]] });
    expect(
      Math.ceil(smpPredict({ smp, inputTensorLike: [[4]] })[0][0])
    ).toBe(5);
    console.log(smpPredict({ smp, inputTensorLike: [[4]] })[0][0]);
  });
});
