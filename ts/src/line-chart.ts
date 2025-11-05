import { addDot, addFrame, defaultMargins, drawLine, getContext } from 'web-ui-common/canvas';
import { removePlaceholder } from 'web-ui-common/dom';
import type { Pair } from 'web-ui-common/types';
import { makeScale } from 'web-ui-common/util';

const colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'tomato'];

export interface LineChartOptions {
  /** Labels for each dataset (for legend) */
  labels?: string[];
  /** Whether to show dots at data points */
  showDots?: boolean;
}

export function initLineChartWidget(
  box: HTMLDivElement,
  dataSets: Pair<number>[][],
  options: LineChartOptions = {}
): void {
  removePlaceholder(box);
  const canvas = document.createElement('canvas');
  canvas.setAttribute('width', '600');
  canvas.setAttribute('height', '300');
  canvas.style.width = '600px';
  canvas.style.height = '300px';
  box.appendChild(canvas);
  initLineChart(
    canvas,
    dataSets,
    options
  );
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
  const { labels } = options;
  if (dataSets.length === 0 || dataSets.every(data => data.length === 0)) {
    return;
  }

  const ctx = getContext(canvas);
  let hoveredPoint: {
    dataSetIndex: number;
    pointIndex: number;
    x: number;
    y: number;
  } | null = null;

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

  function drawChart(): void {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw frame with axes
    addFrame(canvas, defaultMargins, xRange, yRange, 6);

    // Draw each curve
    dataSets.forEach((data, dataSetIndex) => {
      const color = colors[dataSetIndex % colors.length] ?? 'steelblue';

      drawLine(ctx, xScale, yScale, data, {
        stroke: color,
        lineWidth: 2
      });

      // Draw dots at data points if enabled
      if (options.showDots === true) {
        data.forEach(([x, y], pointIndex) => {
          const isHovered = hoveredPoint?.dataSetIndex === dataSetIndex &&
            hoveredPoint.pointIndex === pointIndex;

          addDot(ctx, xScale(x), yScale(y), isHovered ? 6 : 3, color);
        });
      }
    });

    // Draw legend if labels are provided
    if (labels && labels.length > 0) {
      const legendX = canvas.width - defaultMargins.right - 160;
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

    // Draw hover tooltip
    if (hoveredPoint) {
      const data = dataSets[hoveredPoint.dataSetIndex];
      const point = data[hoveredPoint.pointIndex];
      const [x, y] = point;
      const canvasX = xScale(x);
      const canvasY = yScale(y);

      const text = `(${x.toFixed(2)}, ${y.toFixed(2)})`;
      ctx.font = '12px sans-serif';
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(canvasX + 10, canvasY - 25, ctx.measureText(text).width + 8, 20);
      ctx.fillStyle = 'white';
      ctx.fillText(text, canvasX + 14, canvasY - 15);
    }
  }

  function findClosestPoint(
    mouseX: number,
    mouseY: number
  ): { dataSetIndex: number; pointIndex: number; distance: number } | null {
    let closestPoint: { dataSetIndex: number; pointIndex: number; distance: number } | null = null;
    const maxDistance = 20; // pixels

    dataSets.forEach((data, dataSetIndex) => {
      data.forEach(([x, y], pointIndex) => {
        const canvasX = xScale(x);
        const canvasY = yScale(y);
        const distance = Math.sqrt((mouseX - canvasX) ** 2 + (mouseY - canvasY) ** 2);

        if (distance <= maxDistance && (!closestPoint || distance < closestPoint.distance)) {
          closestPoint = { dataSetIndex, pointIndex, distance };
        }
      });
    });

    return closestPoint;
  }

  function handleMouseMove(event: MouseEvent): void {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const closest = findClosestPoint(mouseX, mouseY);

    if (closest !== null) {
      hoveredPoint = {
        dataSetIndex: closest.dataSetIndex,
        pointIndex: closest.pointIndex,
        x: mouseX,
        y: mouseY
      };
      canvas.style.cursor = 'pointer';
    } else {
      hoveredPoint = null;
      canvas.style.cursor = 'default';
    }

    drawChart();
  }

  function handleMouseLeave(): void {
    hoveredPoint = null;
    canvas.style.cursor = 'default';
    drawChart();
  }

  // Add event listeners
  canvas.addEventListener('mousemove', handleMouseMove);
  canvas.addEventListener('mouseleave', handleMouseLeave);

  // Initial draw
  drawChart();
}
