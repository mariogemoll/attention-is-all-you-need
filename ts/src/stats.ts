import type { Pair } from 'web-ui-common/types';

/**
 * Compute exponential moving average of values
 * @param values - Array of numeric values
 * @param alpha - Smoothing factor (0-1), lower = smoother. Typical: 0.1-0.3
 * @returns Smoothed values
 */
export function exponentialMovingAverage(values: number[], alpha = 0.2): number[] {
  const smoothed: number[] = [];
  let current = values[0] ?? 0;

  for (const value of values) {
    current = alpha * value + (1 - alpha) * current;
    smoothed.push(current);
  }

  return smoothed;
}

/**
 * Apply exponential moving average smoothing to [x, y] data
 * @param data - Array of [x, y] pairs
 * @param alpha - Smoothing factor (0-1), lower = smoother. Typical: 0.1-0.3
 * @returns Smoothed [x, y] pairs with original x values and smoothed y values
 */
export function smoothData(data: Pair<number>[], alpha = 0.1): Pair<number>[] {
  const yValues = data.map(([, y]) => y);
  const smoothedY = exponentialMovingAverage(yValues, alpha);
  return data.map(([x], i) => [x, smoothedY[i]]);
}
