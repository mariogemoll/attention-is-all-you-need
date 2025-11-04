import { addFrame, defaultMargins, drawLine, getContext } from 'web-ui-common/canvas';
import type { Pair } from 'web-ui-common/types';
import { makeScale } from 'web-ui-common/util';

import { exponentialMovingAverage } from './stats';

const colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'tomato'];

export interface LossWidgetOptions {
  /** Labels for each dataset (for legend) */
  labels?: string[];
  /** Whether to apply smoothing (default: true) */
  smooth?: boolean;
  /** Smoothing factor (default: 0.1) */
  alpha?: number;
}

/**
 * Initialize a loss curve widget on a canvas
 * @param canvas - Canvas element to render on
 * @param lossDataSets - Array of loss data arrays, each containing [step, loss] pairs
 * @param options - Optional configuration
 */
export function initLossWidget(
  canvas: HTMLCanvasElement,
  lossDataSets: Pair<number>[][],
  options: LossWidgetOptions = {}
): void {
  const { labels, smooth = true, alpha = 0.1 } = options;
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

    // Apply smoothing if enabled
    let dataToPlot: Pair<number>[];
    if (smooth) {
      const smoothedLosses = exponentialMovingAverage(losses, alpha);
      dataToPlot = steps.map((step, i) => [step, smoothedLosses[i]]);
    } else {
      dataToPlot = lossData;
    }

    drawLine(ctx, xScale, yScale, dataToPlot, {
      stroke: color,
      lineWidth: 2
    });
  });

  // Draw legend if labels are provided
  if (labels && labels.length > 0) {
    const legendX = canvas.width - defaultMargins.right - 100;
    const legendY = defaultMargins.top + 20;

    ctx.font = '14px sans-serif';
    ctx.textAlign = 'left';

    labels.forEach((label, index) => {
      const color = colors[index % colors.length] ?? 'steelblue';
      const y = legendY + index * 20;

      ctx.fillStyle = color;
      ctx.fillRect(legendX, y - 8, 20, 3);
      ctx.fillStyle = '#333';
      ctx.fillText(label, legendX + 25, y);
    });
  }
}
