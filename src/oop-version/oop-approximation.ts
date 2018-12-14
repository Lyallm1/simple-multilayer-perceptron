import { MLPerceptron } from './oop-mlp';

const p = new MLPerceptron(
  [[0], [1], [2], [3], [4]],
  [[1, 2, 3, 4, 5]],
  5,
  'sgd',
  'sigmoid',
  'linear',
  'meanSquaredError',
  0.25
);

p.earlyStoppingTraining(2000, 0.000001, 0).then(h => {
  p.predict([[0]]);
  p.predict([[1]]);
  p.predict([[2]]);
  p.predict([[3]]);
  p.predict([[4]]);
});
