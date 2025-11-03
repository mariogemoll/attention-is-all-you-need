import { addFrame, defaultMargins, drawLine, getContext } from 'web-ui-common/canvas';
import type { Pair } from 'web-ui-common/types';
import { makeScale } from 'web-ui-common/util';

/**
 * Initialize a loss curve widget on a canvas
 * @param canvas - Canvas element to render on
 * @param lossData - Array of [epoch, loss] pairs
 */
export function initLossWidget(canvas: HTMLCanvasElement, lossData: Pair<number>[]): void {
  if (lossData.length === 0) {
    return;
  }

  const ctx = getContext(canvas);

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Calculate data ranges
  const epochs = lossData.map(([epoch]) => epoch);
  const losses = lossData.map(([, loss]) => loss);

  const minEpoch = Math.min(...epochs);
  const maxEpoch = Math.max(...epochs);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);

  // Add some padding to the ranges
  const epochRange: Pair<number> = [minEpoch, maxEpoch];
  const lossRange: Pair<number> = [
    minLoss - (maxLoss - minLoss) * 0.1,
    maxLoss + (maxLoss - minLoss) * 0.1
  ];

  // Create scales
  const xScale = makeScale(
    epochRange,
    [defaultMargins.left, canvas.width - defaultMargins.right]
  );

  const yScale = makeScale(
    lossRange,
    [canvas.height - defaultMargins.bottom, defaultMargins.top]
  );

  // Draw frame with axes
  addFrame(canvas, defaultMargins, epochRange, lossRange, 6);

  // Draw loss curve
  drawLine(ctx, xScale, yScale, lossData, {
    stroke: 'steelblue',
    lineWidth: 2
  });
}
