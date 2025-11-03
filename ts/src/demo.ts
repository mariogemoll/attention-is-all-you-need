import { el } from 'web-ui-common/dom';

import { initLossWidget } from './loss';

// Get the canvas element
const canvas = el(document, '#canvas') as HTMLCanvasElement;

// Sample loss data for demonstration
const lossData: [number, number][] = [
  [0, 2.5],
  [1, 2.1],
  [2, 1.8],
  [3, 1.5],
  [4, 1.3],
  [5, 1.1],
  [6, 0.9],
  [7, 0.8],
  [8, 0.7],
  [9, 0.65],
  [10, 0.6]
];

// Initialize the loss widget
initLossWidget(canvas, lossData);
