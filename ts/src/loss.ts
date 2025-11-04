import { addFrame, defaultMargins, drawLine, getContext } from 'web-ui-common/canvas';
import type { Pair } from 'web-ui-common/types';
import { makeScale } from 'web-ui-common/util';

import { exponentialMovingAverage } from './stats';

const colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'tomato'];

/**
 * Initialize a loss curve widget on a canvas
 * @param canvas - Canvas element to render on
 * @param lossDataSets - Array of loss data arrays, each containing [step, loss] pairs
 */
export function initLossWidget(
  canvas: HTMLCanvasElement,
  lossDataSets: Pair<number>[][]
): void {
  if (lossDataSets.length === 0 || lossDataSets.every(data => data.length === 0)) {
    return;
  }

  const ctx = getContext(canvas);

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Calculate global data ranges across all datasets
  let minStep = Infinity;
  let maxStep = -Infinity;
  let minLoss = Infinity;
  let maxLoss = -Infinity;

  for (const lossData of lossDataSets) {
    const steps = lossData.map(([step]) => step);
    const losses = lossData.map(([, loss]) => loss);

    minStep = Math.min(minStep, ...steps);
    maxStep = Math.max(maxStep, ...steps);
    minLoss = Math.min(minLoss, ...losses);
    maxLoss = Math.max(maxLoss, ...losses);
  }

  // Add some padding to the ranges
  const stepRange: Pair<number> = [minStep, maxStep];
  const lossRange: Pair<number> = [
    minLoss - (maxLoss - minLoss) * 0.1,
    maxLoss + (maxLoss - minLoss) * 0.1
  ];

  // Create scales
  const xScale = makeScale(
    stepRange,
    [defaultMargins.left, canvas.width - defaultMargins.right]
  );

  const yScale = makeScale(
    lossRange,
    [canvas.height - defaultMargins.bottom, defaultMargins.top]
  );

  // Draw frame with axes
  addFrame(canvas, defaultMargins, stepRange, lossRange, 6);

  // Draw each loss curve
  lossDataSets.forEach((lossData, index) => {
    const color = colors[index % colors.length] ?? 'steelblue';
    const steps = lossData.map(([step]) => step);
    const losses = lossData.map(([, loss]) => loss);

    // Compute and draw smoothed loss curve
    const smoothedLosses = exponentialMovingAverage(losses, 0.1);
    const smoothedData: Pair<number>[] = steps.map((step, i) => [
      step,
      smoothedLosses[i]
    ]);

    drawLine(ctx, xScale, yScale, smoothedData, {
      stroke: color,
      lineWidth: 2
    });
  });
}
