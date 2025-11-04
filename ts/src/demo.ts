import { el } from 'web-ui-common/dom';

import { setUpBuckets } from './buckets';
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


    const corpora = [
      { label: 'europarl v7', numEntries: 1920209 },
      { label: 'commoncrawl', numEntries: 2399123 },
      { label: 'news commentary v9', numEntries: 201288 },
      { label: 'newstest 2013', numEntries: 3000 },
      { label: 'newstest 2014', numEntries: 3003 }
    ];

    const bucketSizes =  [
      744850, 1915390, 1101320, 435248, 157047, 56598, 21731, 8917, 4029, 2215, 1172
    ];

    const bucketSeqLenStepSize = 16;

    const batchSize = 1000;

    const corporaBox = el(document, '#corpora-widget') as HTMLDivElement;
    const bucketsBox = el(document, '#buckets-widget') as HTMLDivElement;

    setUpBuckets(corporaBox, 1000, corpora, { minLabelPixelWidth: 50 });

    const buckets = bucketSizes.map((d, i) => {
      const start = i * bucketSeqLenStepSize + 1;
      const end = (i + 1) * bucketSeqLenStepSize;
      return { label: `${start.toString()}–${end.toString()}`, numEntries: d };
    });
    setUpBuckets(bucketsBox, batchSize, buckets, { minLabelPixelWidth: 44 });


  } catch (error) {
    console.error('Failed to load loss data:', error);
  }
}

// Initialize the demo
void initDemo();
