import {
  tensor1d,
  tensor2d,
  sequential,
  Sequential,
  layers,
  oneHot,
  train,
  nextFrame,
  tidy,
  Tensor,
  Rank
} from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

interface CreateClassifierOptions {
  inputShape: any[];
  learningRate?: number;
  hiddenUnits?: number;
  outputUnits: number;
  numHiddenLayers?: number;
  hiddenActivationFunction?: string;
}

export const createClassifier = ({
  inputShape,
  learningRate = 0.25,
  hiddenUnits = 16,
  outputUnits,
  numHiddenLayers = 1,
  hiddenActivationFunction = 'tanh'
}: CreateClassifierOptions) => {
  if (numHiddenLayers < 1) {
    throw new Error('numHiddenLayers must be >= 1');
  }
  const classifierModel = sequential();
  const hidden = layers.dense({
    units: hiddenUnits,
    inputShape,
    activation: hiddenActivationFunction
  });
  classifierModel.add(hidden);
  for (let i = 1; i < numHiddenLayers; i++) {
    const extraHidden = layers.dense({
      units: hiddenUnits,
      activation: hiddenActivationFunction
    });
    classifierModel.add(extraHidden);
  }
  const output = layers.dense({
    units: outputUnits,
    activation: 'softmax'
  });
  classifierModel.add(output);

  const optimizer = train.sgd(learningRate);

  classifierModel.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return classifierModel;
};

interface TrainClassifierOptions {
  classifierModel: Sequential;
  inputs: any[];
  outputUnits: number;
  labels: number[];
  epochs?: number;
}

export async function trainClassifier({
  classifierModel,
  inputs,
  outputUnits,
  labels,
  epochs = 100
}: TrainClassifierOptions) {
  const xs = tensor2d(inputs);

  const labelsTensor = tensor1d(labels, 'int32');

  const ys = oneHot(labelsTensor, outputUnits).cast('float32');
  labelsTensor.dispose();

  // This is leaking https://github.com/tensorflow/tfjs/issues/457
  await classifierModel.fit(xs, ys, {
    shuffle: true,
    validationSplit: 0.1,
    epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(epoch);
        console.log('loss: ' + logs.loss.toFixed(5));
        const p = new Promise<void>((resolve, reject) => {
          resolve();
        });
        return p;
      },
      onBatchEnd: async (batch, logs) => {
        await nextFrame();
      },
      onTrainEnd: () => {
        const p = new Promise<void>((resolve, reject) => {
          resolve();
        });
        console.log('finished');
        return p;
      }
    }
  });
}

interface ClassifierPredictOptions {
  classifierModel: Sequential;
  labelList: string[];
  input: number[];
}
export const classifierPredict = async ({
  classifierModel,
  labelList,
  input
}: ClassifierPredictOptions) => {
  const p = new Promise<string>((resolve, reject) => {
    tidy(() => {
      const inputTensor = tensor2d([input]);
      const results = classifierModel.predict(inputTensor) as Tensor<Rank>;
      const resultsArray = Array(results.dataSync())[0] as Float32Array;
      resultsArray.forEach((x, i) => {
        console.log(`${i}: ${x.toFixed(3)} ${labelList[i]}`);
      });

      const argMax = results.argMax(1);
      const index = argMax.dataSync()[0];
      const label = labelList[index];
      resolve(label);
    });
  });
  return p;
};
