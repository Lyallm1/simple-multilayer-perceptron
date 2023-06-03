import '@tensorflow/tfjs-node';

import * as tf from '@tensorflow/tfjs';

export async function trainClassifier({ classifierModel, inputs, outputUnits, labels, epochs = 100 }: {
  classifierModel: tf.Sequential, inputs: any[], outputUnits: number, labels: number[], epochs?: number
}) {
  await classifierModel.fit(tf.tensor2d(inputs), tf.oneHot(tf.tensor1d(labels, 'int32'), outputUnits).cast('float32'), {
    shuffle: true, validationSplit: 0.1, epochs, callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(epoch);
        console.log('loss: ' + logs.loss.toFixed(5));
        return new Promise<void>(resolve => resolve());
      }, onBatchEnd: tf.nextFrame, onTrainEnd: () => {
        const p = new Promise<void>(resolve => resolve());
        console.log('finished');
        return p;
      }
    }
  });
}

export const createClassifier = ({ inputShape, learningRate = 0.25, hiddenUnits = 16, outputUnits, numHiddenLayers = 1, hiddenActivationFunction = 'tanh' }: {
  inputShape: any[], learningRate?: number, hiddenUnits?: number, outputUnits: number, numHiddenLayers?: number, hiddenActivationFunction?: string
}) => {
  if (numHiddenLayers < 1) throw new Error('numHiddenLayers must be >= 1');
  const classifierModel = tf.sequential();
  classifierModel.add(tf.layers.dense({ units: hiddenUnits, inputShape, activation: hiddenActivationFunction }));
  for (let i = 1; i < numHiddenLayers; i++) classifierModel.add(tf.layers.dense({ units: hiddenUnits, activation: hiddenActivationFunction }));
  classifierModel.add(tf.layers.dense({ units: outputUnits, activation: 'softmax' }));
  classifierModel.compile({ optimizer: tf.train.sgd(learningRate), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return classifierModel;
}, classifierPredict = async ({ classifierModel, labelList, input }: { classifierModel: tf.Sequential, labelList: string[], input: number[] }) => new Promise<string>(resolve => tf.tidy(() => {
  const results = classifierModel.predict(tf.tensor2d([input])) as tf.Tensor;
  (Array(results.dataSync())[0] as Float32Array).forEach((x, i) => console.log(`${i}: ${x.toFixed(3)} ${labelList[i]}`));
  resolve(labelList[results.argMax(1).dataSync()[0]]);
}));
