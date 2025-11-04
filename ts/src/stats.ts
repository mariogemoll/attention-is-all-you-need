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
