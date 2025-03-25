/**
 * Draw a Scatter Plot visualization.
 * 
 * @param {string} canvasId The ID of the canvas that this scatter plot
 *      should be drawn to.
 * @param {Object} scatterPlotPlots The data of the plots that should be
 *      plotted. Each element of the list should be another list where
 *      the first element is the x position and the second element is the
 *      y position.
 */
function drawScatterPlot(canvasId, scatterPlotPlots) {
    const SCATTER_PLOT_CANVAS = document.getElementById(canvasId);
    const SCATTER_PLOT_CTX = SCATTER_PLOT_CANVAS.getContext('2d');

    const SCATTER_PLOT_STROKE_COLOR = 'rgb(222, 222, 222)';
    const SCATTER_PLOT_PLOT_RADIUS = 3;
    const SCATTER_PLOT_AXIS_PADDING = 17;

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

    drawAxes(SCATTER_PLOT_CANVAS, SCATTER_PLOT_CTX, SCATTER_PLOT_STROKE_COLOR,
        maxX, maxY, SCATTER_PLOT_AXIS_PADDING);

    SCATTER_PLOT_CTX.strokeStyle = SCATTER_PLOT_STROKE_COLOR;
    SCATTER_PLOT_CTX.beginPath();
    for (plot of scatterPlotPlots) {
        const TRANSFORMED = graphToScreen(SCATTER_PLOT_CANVAS, SCATTER_PLOT_AXIS_PADDING,
            plot[0], plot[1], maxX, maxY);

        SCATTER_PLOT_CTX.moveTo(TRANSFORMED.x + SCATTER_PLOT_PLOT_RADIUS, TRANSFORMED.y);
        SCATTER_PLOT_CTX.arc(TRANSFORMED.x, TRANSFORMED.y,
            SCATTER_PLOT_PLOT_RADIUS, 0, 2 * Math.PI);
    }

    SCATTER_PLOT_CTX.stroke();

    enableResizing(SCATTER_PLOT_CANVAS, function() {
        drawScatterPlot(canvasId, scatterPlotPlots);
    });
}

function conciseString(num) {
    const SCATTER_PLOT_MAXIMUM_NUMBER_LENGTH = 4;
    const SCATTER_PLOT_EXPONENT_FRACTION_DIGITS = 2;
    const s = num.toString();

    if (num.toString().length <= SCATTER_PLOT_MAXIMUM_NUMBER_LENGTH) {
        return s;
    }

    return num.toExponential(SCATTER_PLOT_EXPONENT_FRACTION_DIGITS);
}

/**
 * Draw a set of axes that stretch from -maxX to maxX on the
 * x-axis and -maxY to maxY on the y-axis.
 * 
 * @param {Object} canvas The canvas that these axes should be
 *      drawn to.
 * @param {Object} ctx The context that should be used for
 *      drawing.
 * @param {string} color The color that should be used for
 *      drawing these axes.
 * @param {number} maxX The maximum x position out of all the points
 *      that need to be plotted. This will be used to determine how
 *      many points should be shown on the x-axis.
 * @param {number} maxY The maximum y position out of all the points
 *      that need to be plotted. This will be used to determine how
 *      many points should be shown on the y-axis.
 * @param {number} padding The padding between the end of the canvas
 *      and the axes.
 */
function drawAxes(canvas, ctx, color, maxX, maxY, padding) {
    const SCATTER_PLOT_STEP_COUNT = 20;
    const SCATTER_PLOT_MARKING_LENGTH = 5;
    const SCATTER_PLOT_MARKINGS_MATRIX = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ];
    const AXIS_START_POINT_X = padding;
    const AXIS_START_POINT_Y = canvas.height - padding;
    const AXIS_END_POINT_X = canvas.width - padding;
    const AXIS_END_POINT_Y = padding;
    const SCATTER_PLOT_X_AXIS_Y = canvas.height / 2;
    const SCATTER_PLOT_Y_AXIS_X = canvas.width / 2;

    ctx.strokeStyle = color;
    ctx.beginPath();

    // X Axis
    ctx.moveTo(AXIS_START_POINT_X, SCATTER_PLOT_X_AXIS_Y);
    ctx.lineTo(AXIS_END_POINT_X, SCATTER_PLOT_X_AXIS_Y);

    // Y Axis
    ctx.moveTo(SCATTER_PLOT_Y_AXIS_X, AXIS_START_POINT_Y);
    ctx.lineTo(SCATTER_PLOT_Y_AXIS_X, AXIS_END_POINT_Y);
    
    var stepX = (canvas.width - padding*2) / SCATTER_PLOT_STEP_COUNT;
    var stepY = (canvas.height - padding*2) / SCATTER_PLOT_STEP_COUNT;

    for (var i = 0; i < SCATTER_PLOT_STEP_COUNT; i++) {
        const X = i * stepX;
        const Y = i * stepY;

        ctx.fontStyle = "5px DidactGothic";
        ctx.fillStyle = color;

        // Draw markings
        for (const multipliers of SCATTER_PLOT_MARKINGS_MATRIX) {
            const coords = screenToGraph(canvas, padding, multipliers[0], multipliers[1], maxX, maxY);
            const coordsStr = conciseString((multipliers[0] > 0) ? coords.x : coords.y);
            const measurements = ctx.measureText(coordsStr);
            const startX = SCATTER_PLOT_Y_AXIS_X + X*multipliers[0];
            const startY = SCATTER_PLOT_X_AXIS_Y + Y*multipliers[1];
            const isX = multipliers[0] != 0;

            ctx.moveTo(startX, startY);
            ctx.lineTo(startX + (isX ? 0 : SCATTER_PLOT_MARKING_LENGTH),
                startY + (isX ? SCATTER_PLOT_MARKING_LENGTH : 0));

            ctx.save();
            ctx.translate(
                startX + (isX ? 0 : SCATTER_PLOT_MARKING_LENGTH +
                    measurements.actualBoundingBoxAscent + measurements.actualBoundingBoxDescent),
                startY + (isX ? SCATTER_PLOT_MARKING_LENGTH +
                    measurements.actualBoundingBoxAscent + measurements.actualBoundingBoxDescent : 0)
            );

            if (isX) {
                ctx.rotate(Math.PI / 2)
            }

            ctx.fillText(coordsStr, 0, 0);
            ctx.restore();
        }
    }

    ctx.stroke();
}

/**
 * Convert a position from "graph space" (i.e. its position in
 * mathematical 2D space) to its "screen space" (i.e the actual
 * position that it should be drawn on the canvas).
 * 
 * @param {Object} canvas The canvas that will be used for calculating
 *      the screen co-ordinates.
 * @param {number} padding The padding between the edges of the canvas
 *      and the axes.
 * @param {number} x The x position of the point in "graph space".
 * @param {number} y The y position of the point in "graph space".
 * @param {number} maxX The maximum x position of any point that needs
 *      to be plotted.
 * @param {number} maxY The maximum y position of any point that needs
 *      to be plotted.
 * @returns An object where x is the x position of the point in
 * "screen space" and y is the y position of the point in
 * "screen space".
 */
function graphToScreen(canvas, padding, x, y, maxX, maxY) {
    // Scale such that it can fit on the graph
    x = (x / maxX) * (canvas.width - padding*2);
    y = (y / maxY) * (canvas.height - padding*2);

    // Shift such that (0, 0) -> (width / 2, height / 2)
    x += canvas.width / 2;
    y += canvas.height / 2;

    // Invert y so that it is in the proper place in the canvas
    y = canvas.height - y;

    return {x: x, y: y};
}

/**
 * Convert a position from "screen space" (i.e. its position where it
 * is drawn on the canvas) to its "graph space" (i.e. the actual point
 * in 2D space that the drawn point represents).
 * 
 * @param {Object} canvas The canvas that will be used for calculating
 *      the graph co-ordinates.
 * @param {number} padding The padding between the edge of the canvas
 *      and the axes.
 * @param {number} x The x position of the point in "screen space".
 * @param {number} y The y position of the point in "screen space".
 * @param {number} maxX The maximum x position of any point that needs to be
 *      plotted.
 * @param {number} maxY The maximum y position of any point that needs to be
 *      plotted.
 * @returns An object where x is the x position of the point in
 * "graph space" and y is the y position of the point in
 * "graph space".
 */
function screenToGraph(canvas, padding, x, y, maxX, maxY) {
    // Inverse invert
    y -= canvas.height;
    y *= -1;

    // Inverse shift
    x -= canvas.width / 2;
    y -= canvas.height / 2;

    // Inverse scale
    x /= canvas.width - padding*2;
    y /= canvas.height - padding*2;
    x *= maxX;
    y *= maxY;

    return {x: x, y: y};
}