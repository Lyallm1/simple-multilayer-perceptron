import {
  createClassifier,
  trainClassifier,
  classifierPredict
} from '../classifier';
import * as colorsJson from './colorData.json';

const labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
];

const normalizeInput = (x: number) => x / 255;

test('smp early stopping prediction', () => {
  const setup = async () => {
    const colors = [];
    const labels = [];
    for (const record of colorsJson.entries) {
      const { r, g, b } = record;
      const col = [r, g, b].map(normalizeInput);
      colors.push(col);
      labels.push(labelList.indexOf(record.label));
    }

    const classifierModel = createClassifier({
      inputShape: [3]
    });

    await trainClassifier({
      classifierModel,
      inputs: colors,
      labels,
      epochs: 10
    });

    const unormalizedInput = [255, 21, 60];
    const input = unormalizedInput.map(normalizeInput);

    const outputLabel = await classifierPredict({
      classifierModel,
      labelList,
      input
    });

    expect(outputLabel).toBe('red-ish');
  };

  setup();
});
