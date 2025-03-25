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
    const RECT = SCATTER_PLOT_CANVAS.getBoundingClientRect();
    const SCATTER_PLOT_STROKE_COLOR = 'rgb(222, 222, 222)';
    const SCATTER_PLOT_PLOT_RADIUS = 7;
    const SCATTER_PLOT_AXIS_PADDING = 44;
    const TOOLTIP_SHOW_DISTANCE = SCATTER_PLOT_PLOT_RADIUS + 5;

    SCATTER_PLOT_CTX.clearRect(0, 0, SCATTER_PLOT_CANVAS.width, SCATTER_PLOT_CANVAS.height);

    // Find maximum x and y values
    var maxX = -1;
    var maxY = -1;
    var minX = -1;
    var minY = -1;

    for (plot of scatterPlotPlots) {
        const X = plot[0];
        const Y = plot[1];

        if (maxX == -1 || X > maxX) {
            maxX = X;
        }

        if (maxY == -1 || Y > maxY) {
            maxY = Y;
        }

        if (minX == -1 || X < minX) {
            minX = X;
        }

        if (minY == -1 || Y < minY) {
            minY = Y;
        }
    }

    var transformedPlots = [];

    const SCATTER_PLOT_STEP_COUNT = 20;
    const SCATTER_PLOT_MARKING_LENGTH = 5;
    const SCATTER_PLOT_MARKINGS_MATRIX = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ];
    const AXIS_START_POINT_X = SCATTER_PLOT_AXIS_PADDING;
    const AXIS_START_POINT_Y = SCATTER_PLOT_CANVAS.height - SCATTER_PLOT_AXIS_PADDING;
    const AXIS_END_POINT_X = SCATTER_PLOT_CANVAS.width - SCATTER_PLOT_AXIS_PADDING;
    const AXIS_END_POINT_Y = SCATTER_PLOT_AXIS_PADDING;
    const SCATTER_PLOT_X_AXIS_Y = AXIS_START_POINT_Y;
    const SCATTER_PLOT_Y_AXIS_X = AXIS_START_POINT_X;

    SCATTER_PLOT_CTX.strokeStyle = SCATTER_PLOT_STROKE_COLOR;
    SCATTER_PLOT_CTX.beginPath();

    // X Axis
    SCATTER_PLOT_CTX.moveTo(AXIS_START_POINT_X, SCATTER_PLOT_X_AXIS_Y);
    SCATTER_PLOT_CTX.lineTo(AXIS_END_POINT_X, SCATTER_PLOT_X_AXIS_Y);

    // Y Axis
    SCATTER_PLOT_CTX.moveTo(SCATTER_PLOT_Y_AXIS_X, AXIS_START_POINT_Y);
    SCATTER_PLOT_CTX.lineTo(SCATTER_PLOT_Y_AXIS_X, AXIS_END_POINT_Y);
    SCATTER_PLOT_CTX.stroke();

    var stepX = niceStep(maxX - minX, SCATTER_PLOT_STEP_COUNT);
    var stepY = niceStep(maxY - minY, SCATTER_PLOT_STEP_COUNT);

    // Set the minimum to the closest step equal to
    // or below it.
    minX = stepX*Math.floor(minX/stepX);
    minY = stepY*Math.floor(minY/stepY);

    // Adjust maximum to be SCATTER_PLOT_STEP_COUNT
    // steps away from the minimum, with an additional
    // step for safety.
    maxX = minX + stepX*(SCATTER_PLOT_STEP_COUNT+1);
    maxY = minY + stepY*(SCATTER_PLOT_STEP_COUNT+1);

    var xTickPos = minX;
    var yTickPos = minY;

    SCATTER_PLOT_CTX.font = "15px DidactGothic";
    SCATTER_PLOT_CTX.fillStyle = SCATTER_PLOT_STROKE_COLOR;

    // Precompute the transformed plots to use multiple times later
    for (const PLOT of scatterPlotPlots) {
        transformedPlots.push(
            graphToScreen(SCATTER_PLOT_CANVAS, SCATTER_PLOT_AXIS_PADDING, PLOT[0],
                PLOT[1], minX, minY, maxX, maxY
            )
        );
    }

    while (xTickPos <= maxX || yTickPos <= maxY) {
        // Fix potential floating point problems
        xTickPos = parseFloat(xTickPos.toPrecision(12));
        yTickPos = parseFloat(yTickPos.toPrecision(12));

        const SCREEN_COORDS = graphToScreen(SCATTER_PLOT_CANVAS, SCATTER_PLOT_AXIS_PADDING,
            xTickPos, yTickPos, minX, minY, maxX, maxY);
        const MEASUREMENTS_X = SCATTER_PLOT_CTX.measureText(xTickPos.toString());
        const MEASUREMENTS_Y = SCATTER_PLOT_CTX.measureText(yTickPos.toString());
        const TEXT_WIDTH_X = MEASUREMENTS_X.width;
        const TEXT_HEIGHT_X = MEASUREMENTS_X.actualBoundingBoxAscent +
            MEASUREMENTS_X.actualBoundingBoxDescent;
        const TEXT_WIDTH_Y = MEASUREMENTS_Y.width;
        const TEXT_HEIGHT_Y = MEASUREMENTS_Y.actualBoundingBoxAscent +
            MEASUREMENTS_Y.actualBoundingBoxDescent;

        SCATTER_PLOT_CTX.beginPath();

        // X Axis
        if (xTickPos <= maxX) {
            SCATTER_PLOT_CTX.moveTo(SCREEN_COORDS.x, SCATTER_PLOT_X_AXIS_Y);
            SCATTER_PLOT_CTX.lineTo(SCREEN_COORDS.x, SCATTER_PLOT_X_AXIS_Y + SCATTER_PLOT_MARKING_LENGTH);

            SCATTER_PLOT_CTX.fillText(xTickPos.toString(),
                SCREEN_COORDS.x - TEXT_WIDTH_X/2,
                SCATTER_PLOT_X_AXIS_Y + SCATTER_PLOT_MARKING_LENGTH + TEXT_HEIGHT_X);
        }

        // Y Axis
        if (yTickPos <= maxY) {
            SCATTER_PLOT_CTX.moveTo(SCATTER_PLOT_Y_AXIS_X, SCREEN_COORDS.y);
            SCATTER_PLOT_CTX.lineTo(SCATTER_PLOT_Y_AXIS_X - SCATTER_PLOT_MARKING_LENGTH, SCREEN_COORDS.y);

            SCATTER_PLOT_CTX.fillText(yTickPos.toString(),
                SCATTER_PLOT_Y_AXIS_X - SCATTER_PLOT_MARKING_LENGTH*2 - TEXT_WIDTH_Y,
                SCREEN_COORDS.y + TEXT_HEIGHT_Y/2);
        }

        xTickPos += stepX;
        yTickPos += stepY;
        SCATTER_PLOT_CTX.stroke();
    }

    // Draw points
    SCATTER_PLOT_CTX.strokeStyle = SCATTER_PLOT_STROKE_COLOR;
    SCATTER_PLOT_CTX.beginPath();
    for (var i = 0; i < scatterPlotPlots.length; i++) {
        const PLOT = scatterPlotPlots[i];
        const TRANSFORMED = transformedPlots[i];
        const RGB = calculateRgb(1, 2, 0);

        SCATTER_PLOT_CTX.beginPath();
        SCATTER_PLOT_CTX.strokeStyle = RGB;
        SCATTER_PLOT_CTX.fillStyle = RGB;
        SCATTER_PLOT_CTX.arc(TRANSFORMED.x,
            TRANSFORMED.y,
            SCATTER_PLOT_PLOT_RADIUS, 0, 2 * Math.PI);
        SCATTER_PLOT_CTX.fill();
        SCATTER_PLOT_CTX.stroke();
    }

    SCATTER_PLOT_CANVAS.onmousemove = function(event) {
        const mouseX = event.clientX - RECT.left;
        const mouseY = event.clientY - RECT.top;

        drawScatterPlot(canvasId, scatterPlotPlots);

        for (var i = 0; i < transformedPlots.length; i++) {
            const TRANSFORMED_PLOT = transformedPlots[i];
            const ORIGINAL_PLOT = scatterPlotPlots[i];
            const ORIGINAL_PLOT_X = ORIGINAL_PLOT[0];
            const ORIGINAL_PLOT_Y = ORIGINAL_PLOT[1];

            if (mouseX >= TRANSFORMED_PLOT.x - TOOLTIP_SHOW_DISTANCE &&
                    mouseX <= TRANSFORMED_PLOT.x + TOOLTIP_SHOW_DISTANCE &&
                    mouseY >= TRANSFORMED_PLOT.y - TOOLTIP_SHOW_DISTANCE &&
                    mouseY <= TRANSFORMED_PLOT.y + TOOLTIP_SHOW_DISTANCE) {
                drawTooltip([[{text: `X: ${ORIGINAL_PLOT_X}`, color: "black"},
                    {text: `Y: ${ORIGINAL_PLOT_Y}`, color: "black"}]],
                    mouseX, mouseY, 200, 200, 15, SCATTER_PLOT_CTX);
            }
        }
    }

    enableResizing(SCATTER_PLOT_CANVAS, function() {
        drawScatterPlot(canvasId, scatterPlotPlots);
    });
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
 * @param {number} minX The minimum x position of any point that needs
 *      to be plotted.
 * @param {number} minY The maximum y position of any point that needs
 *      to be plotted.
 * @param {number} maxX The maximum x position of any point that needs
 *      to be plotted.
 * @param {number} maxY The maximum y position of any point that needs
 *      to be plotted.
 * @returns An object where x is the x position of the point in
 * "screen space" and y is the y position of the point in
 * "screen space".
 */
function graphToScreen(canvas, padding, x, y, minX, minY, maxX, maxY,) {
    // Scale such that it can fit on the graph
    x = ((x - minX) / (maxX - minX)) * (canvas.width - padding*2);
    y = ((y - minY) / (maxY - minY)) * (canvas.height - padding*2);

    // Invert y so that it is in the proper place in the canvas
    y = canvas.height - y;

    return {x: x + padding, y: y - padding};
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
 * @param {number} minX The minimum x position of any point that needs to be
 *      plotted.
 * @param {number} minY The minimum y position of any point that needs to be
 *      plotted.
 * @param {number} maxX The maximum x position of any point that needs to be
 *      plotted.
 * @param {number} maxY The maximum y position of any point that needs to be
 *      plotted.
 * @returns An object where x is the x position of the point in
 * "graph space" and y is the y position of the point in
 * "graph space".
 */
function screenToGraph(canvas, padding, x, y, minX, minY, maxX, maxY) {
    x -= padding;
    y += padding;

    // Inverse invert
    y -= canvas.height;
    y *= -1;

    // Inverse scale
    x /= canvas.width - padding*2;
    y /= canvas.height - padding*2;
    x *= (x - minX) / (maxX - minX);
    y *= (y - minY) / (maxY - minY);

    return {x: x, y: y};
}