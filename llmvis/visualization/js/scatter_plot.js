const SCATTER_PLOT_CANVAS = document.getElementById('llmvis-scatterplot-canvas');
const SCATTER_PLOT_CTX = SCATTER_PLOT_CANVAS.getContext('2d');

const SCATTER_PLOT_STROKE_COLOR = 'rgb(222, 222, 222)';
const SCATTER_PLOT_AXIS_PADDING = 17;
const AXIS_START_POINT_X = SCATTER_PLOT_AXIS_PADDING;
const AXIS_START_POINT_Y = SCATTER_PLOT_CANVAS.height - SCATTER_PLOT_AXIS_PADDING;
const AXIS_END_POINT_X = SCATTER_PLOT_CANVAS.width - SCATTER_PLOT_AXIS_PADDING;
const AXIS_END_POINT_Y = SCATTER_PLOT_AXIS_PADDING;
const SCATTER_PLOT_X_AXIS_Y = SCATTER_PLOT_CANVAS.height / 2;
const SCATTER_PLOT_Y_AXIS_X = SCATTER_PLOT_CANVAS.width / 2;
const SCATTER_PLOT_PLOT_RADIUS = 3;
const SCATTER_PLOT_STEP_COUNT = 20;
const SCATTER_PLOT_MARKING_LENGTH = 5;
const SCATTER_PLOT_MAXIMUM_NUMBER_LENGTH = 4;
const SCATTER_PLOT_EXPONENT_FRACTION_DIGITS = 2;
const SCATTER_PLOT_MARKINGS_MATRIX = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
];

var scatterPlotPlots;

/**
 * Draw a Scatter Plot visualization.
 */
function drawScatterPlot() {
    // Find maximum x and y values
    var maxX = -1;
    var maxY = -1;

    for (plot of scatterPlotPlots) {
        const X = Math.abs(plot[0]);
        const Y = Math.abs(plot[1]);

        if (X > maxX) {
            maxX = X;
        }

        if (Y > maxY) {
            maxY = Y;
        }
    }

    // Multiply by 2 to account for both
    // positive and negative values
    maxX *= 2;
    maxY *= 2;

    drawAxes(maxX, maxY);

    SCATTER_PLOT_CTX.strokeStyle = SCATTER_PLOT_STROKE_COLOR;
    SCATTER_PLOT_CTX.beginPath();
    for (plot of scatterPlotPlots) {
        const TRANSFORMED = graphToScreen(plot[0], plot[1], maxX, maxY);

        SCATTER_PLOT_CTX.moveTo(TRANSFORMED.x + SCATTER_PLOT_PLOT_RADIUS, TRANSFORMED.y);
        SCATTER_PLOT_CTX.arc(TRANSFORMED.x, TRANSFORMED.y,
            SCATTER_PLOT_PLOT_RADIUS, 0, 2 * Math.PI);
    }

    SCATTER_PLOT_CTX.stroke();
}

function conciseString(num) {
    const s = num.toString();

    if (num.toString().length <= SCATTER_PLOT_MAXIMUM_NUMBER_LENGTH) {
        return s;
    }

    return num.toExponential(SCATTER_PLOT_EXPONENT_FRACTION_DIGITS);
}

/**
 * Draw a set of axes that stretch from -maxX to maxX on the
 * x-axis and -maxY to maxY on the y-axis.
 * @param {*} maxX The maximum x position out of all the points
 * that need to be plotted. This will be used to determine how
 * many points should be shown on the x-axis.
 * @param {*} maxY The maximum y position out of all the points
 * that need to be plotted. This will be used to determine how
 * many points should be shown on the y-axis.
 */
function drawAxes(maxX, maxY) {
    SCATTER_PLOT_CTX.strokeStyle = SCATTER_PLOT_STROKE_COLOR;
    SCATTER_PLOT_CTX.beginPath();

    // X Axis
    SCATTER_PLOT_CTX.moveTo(AXIS_START_POINT_X, SCATTER_PLOT_X_AXIS_Y);
    SCATTER_PLOT_CTX.lineTo(AXIS_END_POINT_X, SCATTER_PLOT_X_AXIS_Y);

    // Y Axis
    SCATTER_PLOT_CTX.moveTo(SCATTER_PLOT_Y_AXIS_X, AXIS_START_POINT_Y);
    SCATTER_PLOT_CTX.lineTo(SCATTER_PLOT_Y_AXIS_X, AXIS_END_POINT_Y);
    
    var stepX = (SCATTER_PLOT_CANVAS.width - SCATTER_PLOT_AXIS_PADDING*2) / SCATTER_PLOT_STEP_COUNT;
    var stepY = (SCATTER_PLOT_CANVAS.height - SCATTER_PLOT_AXIS_PADDING*2) / SCATTER_PLOT_STEP_COUNT;

    for (i = 0; i < SCATTER_PLOT_STEP_COUNT; i++) {
        const X = i * stepX;
        const Y = i * stepY;

        SCATTER_PLOT_CTX.fontStyle = "5px DidactGothic";
        SCATTER_PLOT_CTX.fillStyle = SCATTER_PLOT_STROKE_COLOR;

        // Draw markings
        for (const multipliers of SCATTER_PLOT_MARKINGS_MATRIX) {
            const coords = screenToGraph(multipliers[0], multipliers[1], maxX, maxY);
            const coordsStr = conciseString((multipliers[0] > 0) ? coords.x : coords.y);
            const measurements = SCATTER_PLOT_CTX.measureText(coordsStr);
            const startX = SCATTER_PLOT_Y_AXIS_X + X*multipliers[0];
            const startY = SCATTER_PLOT_X_AXIS_Y + Y*multipliers[1];
            const isX = multipliers[0] != 0;

            SCATTER_PLOT_CTX.moveTo(startX, startY);
            SCATTER_PLOT_CTX.lineTo(startX + (isX ? 0 : SCATTER_PLOT_MARKING_LENGTH),
                startY + (isX ? SCATTER_PLOT_MARKING_LENGTH : 0));

            SCATTER_PLOT_CTX.save();
            SCATTER_PLOT_CTX.translate(
                startX + (isX ? 0 : SCATTER_PLOT_MARKING_LENGTH +
                    measurements.actualBoundingBoxAscent + measurements.actualBoundingBoxDescent),
                startY + (isX ? SCATTER_PLOT_MARKING_LENGTH +
                    measurements.actualBoundingBoxAscent + measurements.actualBoundingBoxDescent : 0)
            );

            if (isX) {
                SCATTER_PLOT_CTX.rotate(Math.PI / 2)
            }

            SCATTER_PLOT_CTX.fillText(coordsStr, 0, 0);
            SCATTER_PLOT_CTX.restore();
        }
    }

    SCATTER_PLOT_CTX.stroke();
}

/**
 * Convert a position from "graph space" (i.e. its position in
 * mathematical 2D space) to its "screen space" (i.e the actual
 * position that it should be drawn on the canvas).
 * @param {*} x The x position of the point in "graph space".
 * @param {*} y The y position of the point in "graph space".
 * @param {*} maxX The maximum x position of any point that needs
 * to be plotted.
 * @param {*} maxY The maximum y position of any point that needs
 * to be plotted.
 * @returns An object where x is the x position of the point in
 * "screen space" and y is the y position of the point in
 * "screen space".
 */
function graphToScreen(x, y, maxX, maxY) {
    // Scale such that it can fit on the graph
    x = (x / maxX) * (SCATTER_PLOT_CANVAS.width - SCATTER_PLOT_AXIS_PADDING*2);
    y = (y / maxY) * (SCATTER_PLOT_CANVAS.height - SCATTER_PLOT_AXIS_PADDING*2);

    // Shift such that (0, 0) -> (width / 2, height / 2)
    x += SCATTER_PLOT_CANVAS.width / 2;
    y += SCATTER_PLOT_CANVAS.height / 2;

    // Invert y so that it is in the proper place in the canvas
    y = SCATTER_PLOT_CANVAS.height - y;

    return {x: x, y: y};
}

/**
 * Convert a position from "screen space" (i.e. its position where it
 * is drawn on the canvas) to its "graph space" (i.e. the actual point
 * in 2D space that the drawn point represents).
 * @param {*} x The x position of the point in "screen space".
 * @param {*} y The y position of the point in "screen space".
 * @param {*} maxX The maximum x position of any point that needs to be
 * plotted.
 * @param {*} maxY The maximum y position of any point that needs to be
 * plotted.
 * @returns An object where x is the x position of the point in
 * "graph space" and y is the y position of the point in
 * "graph space".
 */
function screenToGraph(x, y, maxX, maxY) {
    // Inverse invert
    y -= SCATTER_PLOT_CANVAS.height;
    y *= -1;

    // Inverse shift
    x -= SCATTER_PLOT_CANVAS.width / 2;
    y -= SCATTER_PLOT_CANVAS.height / 2;

    // Inverse scale
    x /= SCATTER_PLOT_CANVAS.width - SCATTER_PLOT_AXIS_PADDING*2;
    y /= SCATTER_PLOT_CANVAS.height - SCATTER_PLOT_AXIS_PADDING*2;
    x *= maxX;
    y *= maxY;

    return {x: x, y: y};
}