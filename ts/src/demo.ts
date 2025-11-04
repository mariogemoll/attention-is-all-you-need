import { el } from 'web-ui-common/dom';

import { convertToLossData, loadBinaryFloats, loadMultipleLossData } from './data-loader';
import { initLineChart } from './line-chart';
import { smoothData } from './stats';

// Get the canvas elements
const canvas = el(document, '#canvas') as HTMLCanvasElement;
const epochCanvas = el(document, '#epoch-canvas') as HTMLCanvasElement;

// Load and display the loss data
async function initDemo(): Promise<void> {
  try {
    // Show loading message
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#666';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Loading loss data...', canvas.width / 2, canvas.height / 2);
    }

    // Load multiple loss data files
    const lossDataSets = await loadMultipleLossData([
      './training_run_1.loss.bin',
      './training_run_2.loss.bin'
    ]);

    // Smooth the step-level data and display
    const smoothedDataSets = lossDataSets.map(data => smoothData(data, 0.1));
    initLineChart(canvas, smoothedDataSets, {
      labels: ['Run 1', 'Run 2']
    });

    // Load epoch-level loss data
    const epochTrainLoss = await loadBinaryFloats('./epoch_train.loss.bin');
    const epochValLoss = await loadBinaryFloats('./epoch_val.loss.bin');

    // Display epoch data without smoothing
    initLineChart(
      epochCanvas,
      [convertToLossData(epochTrainLoss), convertToLossData(epochValLoss)],
      { labels: ['Train', 'Val'] }
    );
  } catch (error) {
    console.error('Failed to load loss data:', error);

    // Show error message on canvas
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#cc0000';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Error loading loss data', canvas.width / 2, canvas.height / 2 - 10);
      ctx.fillStyle = '#666';
      ctx.font = '12px sans-serif';
      ctx.fillText('Check console for details', canvas.width / 2, canvas.height / 2 + 15);
    }
  }
}

// Initialize the demo
void initDemo();
