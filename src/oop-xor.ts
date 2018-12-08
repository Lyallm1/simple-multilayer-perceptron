import { MLPerceptron } from './oop-mlp';

const p2 = new MLPerceptron(
  [[0, 0], [0, 1], [1, 0], [1, 1]],
  [[0, 1, 1, 0]],
  5,
  'adam',
  'sigmoid',
  'sigmoid',
  'meanSquaredError',
  0.25
);

p2.train(200, 0).then(h => {
  const conf = p2.confMatrix(p2.inputs, p2.targets);
  conf.print();
  console.log('Precision: ' + p2.precision * 100 + '%');
});
