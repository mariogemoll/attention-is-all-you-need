import { addFrame, defaultMargins, drawLine, getContext } from 'web-ui-common/canvas';
import type { Pair } from 'web-ui-common/types';
import { makeScale } from 'web-ui-common/util';

import { exponentialMovingAverage } from './stats';

const colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'tomato'];

export interface LineChartOptions {
  /** Labels for each dataset (for legend) */
  labels?: string[];
  /** Whether to apply smoothing (default: true) */
  smooth?: boolean;
  /** Smoothing factor (default: 0.1) */
  alpha?: number;
}

/**
 * Initialize a line chart on a canvas
 * @param canvas - Canvas element to render on
 * @param dataSets - Array of data arrays, each containing [x, y] pairs
 * @param options - Optional configuration
 */
export function initLineChart(
  canvas: HTMLCanvasElement,
  dataSets: Pair<number>[][],
  options: LineChartOptions = {}
): void {
  const { labels, smooth = true, alpha = 0.1 } = options;
  if (dataSets.length === 0 || dataSets.every(data => data.length === 0)) {
    return;
  }

  const ctx = getContext(canvas);

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Calculate global data ranges across all datasets
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const data of dataSets) {
    const xValues = data.map(([x]) => x);
    const yValues = data.map(([, y]) => y);

    minX = Math.min(minX, ...xValues);
    maxX = Math.max(maxX, ...xValues);
    minY = Math.min(minY, ...yValues);
    maxY = Math.max(maxY, ...yValues);
  }

  // Add some padding to the ranges
  const xRange: Pair<number> = [minX, maxX];
  const yRange: Pair<number> = [
    minY - (maxY - minY) * 0.1,
    maxY + (maxY - minY) * 0.1
  ];

  // Create scales
  const xScale = makeScale(
    xRange,
    [defaultMargins.left, canvas.width - defaultMargins.right]
  );

  const yScale = makeScale(
    yRange,
    [canvas.height - defaultMargins.bottom, defaultMargins.top]
  );

  // Draw frame with axes
  addFrame(canvas, defaultMargins, xRange, yRange, 6);

  // Draw each curve
  dataSets.forEach((data, index) => {
    const color = colors[index % colors.length] ?? 'steelblue';
    const xValues = data.map(([x]) => x);
    const yValues = data.map(([, y]) => y);

    // Apply smoothing if enabled
    let dataToPlot: Pair<number>[];
    if (smooth) {
      const smoothedY = exponentialMovingAverage(yValues, alpha);
      dataToPlot = xValues.map((x, i) => [x, smoothedY[i]]);
    } else {
      dataToPlot = data;
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
