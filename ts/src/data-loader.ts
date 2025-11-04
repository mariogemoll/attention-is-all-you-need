/**
 * Utilities for loading binary data files
 */

/**
 * Download and parse a binary file containing 32-bit floats
 * @param url - URL of the binary file to download
 * @returns Promise that resolves to an array of numbers
 */
export async function loadBinaryFloats(url: string): Promise<number[]> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const float32Array = new Float32Array(arrayBuffer);

    // Convert Float32 values to regular numbers
    return Array.from(float32Array);
  } catch (error) {
    console.error('Error loading binary file:', error);
    throw error;
  }
}

/**
 * Convert a flat array of loss values to [step, loss] pairs
 * @param lossValues - Array of loss values
 * @returns Array of [step, loss] pairs
 */
export function convertToLossData(lossValues: number[]): [number, number][] {
  return lossValues.map((loss, step) => [step, loss]);
}

/**
 * Load loss data from a binary file and convert to chart format
 * @param url - URL of the loss.bin file
 * @returns Promise that resolves to an array of [step, loss] pairs
 */
export async function loadLossData(url: string): Promise<[number, number][]> {
  const lossValues = await loadBinaryFloats(url);
  return convertToLossData(lossValues);
}

/**
 * Load multiple loss data files
 * @param urls - Array of URLs to loss.bin files
 * @returns Promise that resolves to an array of loss data arrays
 */
export async function loadMultipleLossData(
  urls: string[]
): Promise<[number, number][][]> {
  return Promise.all(urls.map(url => loadLossData(url)));
}
