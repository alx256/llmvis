/**
 * Draw a line chart visualization.
 * 
 * @param {string} canvasId The ID of the canvas that this
 *      line chart should be drawn to.
 * @param {Object} lineChartValues A list containing the
 *      values that this line chart should show. Each element
 *      of the list should have the categorical or numerical
 *      data for the x axis as the first element and the
 *      corresponding numerical y axis value as the second.
 * @param {string} xLabel The label that should be shown on
 *      the x-axis. Default is an empty label.
 * @param {string} yLabel The label that should be shown on
 *      the y-axis. Default is an empty label.
 */
function drawLineChart(canvasId, lineChartValues, xLabel, yLabel) {
    const LINE_CHART_CANVAS = document.getElementById(canvasId);
    const LINE_CHART_CTX = LINE_CHART_CANVAS.getContext('2d');
    const RECT = LINE_CHART_CANVAS.getBoundingClientRect();

    const LINE_CHART_STROKE_COLOR = 'rgb(222, 222, 222)';
    const LINE_CHART_AXIS_PADDING = 56;

    // Clear canvas in case line charts have been drawn before
    LINE_CHART_CTX.clearRect(0, 0, LINE_CHART_CANVAS.width, LINE_CHART_CANVAS.height);

    var maxYVal = -1;
    var minYVal = -1;

    const AXIS_START_POINT_X = LINE_CHART_AXIS_PADDING;
    const AXIS_START_POINT_Y = LINE_CHART_CANVAS.height - LINE_CHART_AXIS_PADDING;
    const AXIS_END_POINT_X = LINE_CHART_CANVAS.width - LINE_CHART_AXIS_PADDING;
    const AXIS_END_POINT_Y = LINE_CHART_AXIS_PADDING;
    const Y_TICK_COUNT = 10;
    const POINT_RADIUS = 2;
    const X_GAP = (AXIS_END_POINT_X - AXIS_START_POINT_X) / (lineChartValues.length - 1);
    const POINT_HOVER_HITBOX = 20;
    const EXPLANATION_BOX_WIDTH = 200;
    const EXPLANATION_BOX_HEIGHT = 200;

    var roundDp = 0;
    var last;

    LINE_CHART_CANVAS.onmousemove = function (event) {
        const mouseX = event.clientX - RECT.left;
        const mouseY = event.clientY - RECT.top;

        // Find what we are hovering over
        const GRAPH_X = (mouseX - LINE_CHART_AXIS_PADDING) / X_GAP;
        const i = Math.round(GRAPH_X);

        // Refresh
        drawLineChart(canvasId, lineChartValues, xLabel, yLabel);

        // Vertical line
        LINE_CHART_CTX.strokeStyle = LINE_CHART_STROKE_COLOR;
        LINE_CHART_CTX.moveTo(mouseX, AXIS_END_POINT_Y);
        LINE_CHART_CTX.lineTo(mouseX, AXIS_START_POINT_Y);
        LINE_CHART_CTX.stroke();

        if (lineChartValues[i] == undefined ||
            GRAPH_X % 1 * X_GAP > POINT_HOVER_HITBOX && (1.0 - GRAPH_X % 1) * X_GAP > POINT_HOVER_HITBOX) {
            return;
        }

        // Draw explanation box
        const TEXT = [
            [{ text: `X: ${lineChartValues[i].x} Y: ${lineChartValues[i].y}`, color: 'black' }],
            [{ text: (lineChartValues[i].detail != undefined) ? lineChartValues[i].detail : "", color: 'black' }]
        ];

        drawTooltip(TEXT, mouseX, mouseY, EXPLANATION_BOX_WIDTH, EXPLANATION_BOX_HEIGHT, 12, LINE_CHART_CTX);
    };

    LINE_CHART_CANVAS.onmouseout = function (event) {
        // Refresh
        drawLineChart(canvasId, lineChartValues, xLabel, yLabel);
    };

    /*
    Find the maximum value in the list,
    whether or not the data is integer data or
    real-valued and the number of decimal
    places that values on the x-axis should
    be rounded to.
    */
    for (value of lineChartValues) {
        if (maxYVal == -1 || value.y > maxYVal) {
            maxYVal = value.y;
        }

        if (minYVal == -1 || value.y < minYVal) {
            minYVal = value.y;
        }

        if (last != undefined && last != value.x) {
            while (last.toFixed(roundDp) == value.x.toFixed(roundDp)) {
                roundDp += 1;
            }
        }

        last = value.x;
    }

    // X Axis
    drawAxis(
        LINE_CHART_CTX,
        LINE_CHART_AXIS_PADDING,
        LINE_CHART_AXIS_PADDING,
        LINE_CHART_STROKE_COLOR,
        categoricalData(lineChartValues.map((v) => v.x)),
        AxisPosition.BOTTOM,
        xLabel,
        LocalTickPosition.FULL
    )

    // Y Axis
    const AXIS_DRAW_RESULTS = drawAxis(
        LINE_CHART_CTX,
        LINE_CHART_AXIS_PADDING,
        LINE_CHART_AXIS_PADDING,
        LINE_CHART_STROKE_COLOR,
        continuousData(minYVal, maxYVal, Math.min(Math.ceil(maxYVal - minYVal), Y_TICK_COUNT)),
        AxisPosition.LEFT,
        yLabel
    )

    var maxVal = AXIS_DRAW_RESULTS.max;

    // Gap between each point on the x axis
    var previousX;
    var previousY;
    var nextX;

    LINE_CHART_CTX.strokeStyle = LINE_CHART_STROKE_COLOR;

    // Draw the points and lines themselves
    for (var i = 0; i < lineChartValues.length; i++) {
        const ENTRY = lineChartValues[i];
        const NAME = ENTRY.x;
        const VALUE = ENTRY.y;
        const X = LINE_CHART_AXIS_PADDING + X_GAP * i;
        const Y = AXIS_END_POINT_Y + (1 - (VALUE / maxVal)) * ((AXIS_START_POINT_Y - AXIS_END_POINT_Y));
        const MEASUREMENTS = LINE_CHART_CTX.measureText(NAME.toFixed(roundDp));
        const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent + MEASUREMENTS.actualBoundingBoxDescent;

        // Line from previous point (if applicable)
        if (previousX != undefined && previousY != undefined) {
            LINE_CHART_CTX.moveTo(previousX, previousY);
            LINE_CHART_CTX.lineTo(X, Y);
        }

        LINE_CHART_CTX.moveTo(X + 2, Y);
        // Point (as a filled circle)
        LINE_CHART_CTX.arc(X, Y, POINT_RADIUS, 0, 2 * Math.PI);
        LINE_CHART_CTX.fill();
        // Store previous x and y to draw a line from when drawing the
        // next point.
        previousX = X;
        previousY = Y;

        // Ignore if this will be drawn over the previous x label
        if (nextX != undefined && X < nextX) {
            continue;
        }

        nextX = X + MEASUREMENTS.width;
    }

    LINE_CHART_CTX.stroke();

    enableResizing(LINE_CHART_CANVAS, function() {
        drawLineChart(canvasId, lineChartValues, xLabel, yLabel);
    });
}