/**
 * Draw a Scatter Plot visualization.
 * 
 * @param {string} canvasId The ID of the canvas that this scatter plot
 *      should be drawn to.
 * @param {Object} scatterPlotPlots The data of the plots that should be
 *      plotted. Each element of the list should be another list where
 *      the first element is the x position and the second element is the
 *      y position.
 * @param {string} xLabel The label that should be shown on the x-axis.
 *      Default is an empty label.
 * @param {string} yLabel The label that should be shown on the y-axis.
 *      Default is an empty label.
 */
function drawScatterPlot(canvasId, scatterPlotPlots, xLabel, yLabel) {
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
        if (maxX == -1 || plot.x > maxX) {
            maxX = plot.x;
        }

        if (maxY == -1 || plot.y > maxY) {
            maxY = plot.y;
        }

        if (minX == -1 || plot.x < minX) {
            minX = plot.x;
        }

        if (minY == -1 || plot.y < minY) {
            minY = plot.y;
        }
    }

    var transformedPlots = [];

    const SCATTER_PLOT_STEP_COUNT = 20;
    const SCATTER_PLOT_MARKINGS_MATRIX = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ];

    // X Axis
    const AXES_DRAW_RESULTS_X = drawAxis(
        SCATTER_PLOT_CTX,
        SCATTER_PLOT_AXIS_PADDING,
        SCATTER_PLOT_AXIS_PADDING,
        SCATTER_PLOT_STROKE_COLOR,
        continuousData(minX, maxX, SCATTER_PLOT_STEP_COUNT),
        AxisPosition.BOTTOM,
        xLabel
    );

    // Y Axis
    const AXES_DRAW_RESULTS_Y = drawAxis(
        SCATTER_PLOT_CTX,
        SCATTER_PLOT_AXIS_PADDING,
        SCATTER_PLOT_AXIS_PADDING,
        SCATTER_PLOT_STROKE_COLOR,
        continuousData(minY, maxY, SCATTER_PLOT_STEP_COUNT),
        AxisPosition.LEFT,
        yLabel
    );

    minX = AXES_DRAW_RESULTS_X.min;
    maxX = AXES_DRAW_RESULTS_X.max;
    minY = AXES_DRAW_RESULTS_Y.min;
    maxY = AXES_DRAW_RESULTS_Y.max;

    // Precompute the transformed plots to use multiple times later
    for (const PLOT of scatterPlotPlots) {
        transformedPlots.push(
            graphToScreen(SCATTER_PLOT_CANVAS, SCATTER_PLOT_AXIS_PADDING, PLOT.x,
                PLOT.y, minX, minY, maxX, maxY
            )
        );
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

        drawScatterPlot(canvasId, scatterPlotPlots, xLabel, yLabel);

        for (var i = 0; i < transformedPlots.length; i++) {
            const TRANSFORMED_PLOT = transformedPlots[i];
            const ORIGINAL_PLOT = scatterPlotPlots[i];

            if (mouseX >= TRANSFORMED_PLOT.x - TOOLTIP_SHOW_DISTANCE &&
                    mouseX <= TRANSFORMED_PLOT.x + TOOLTIP_SHOW_DISTANCE &&
                    mouseY >= TRANSFORMED_PLOT.y - TOOLTIP_SHOW_DISTANCE &&
                    mouseY <= TRANSFORMED_PLOT.y + TOOLTIP_SHOW_DISTANCE) {
                drawTooltip([[{text: `X: ${ORIGINAL_PLOT.x}`, color: "black"},
                    {text: `Y: ${ORIGINAL_PLOT.y}`, color: "black"}],
                    [{text: (ORIGINAL_PLOT.detail != undefined) ? ORIGINAL_PLOT.detail : "", color: "black"}]],
                    mouseX, mouseY, 200, 200, 15, SCATTER_PLOT_CTX);
            }
        }
    }

    enableResizing(SCATTER_PLOT_CANVAS, function() {
        drawScatterPlot(canvasId, scatterPlotPlots, xLabel, yLabel);
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