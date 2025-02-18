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
 */
function drawLineChart(canvasId, lineChartValues) {
    const LINE_CHART_CANVAS = document.getElementById(canvasId);
    const LINE_CHART_CTX = LINE_CHART_CANVAS.getContext('2d');
    const RECT = LINE_CHART_CANVAS.getBoundingClientRect();

    const LINE_CHART_STROKE_COLOR = 'rgb(222, 222, 222)';
    const LINE_CHART_AXIS_PADDING = 56;

    // Clear canvas in case line charts have been drawn before
    LINE_CHART_CTX.clearRect(0, 0, LINE_CHART_CANVAS.width, LINE_CHART_CANVAS.height);

    var maxVal = -1;
    var minVal = -1;
    var intValues = true;

    const AXIS_START_POINT_X = LINE_CHART_AXIS_PADDING;
    const AXIS_START_POINT_Y = LINE_CHART_CANVAS.height - LINE_CHART_AXIS_PADDING;
    const AXIS_END_POINT_X = LINE_CHART_CANVAS.width - LINE_CHART_AXIS_PADDING;
    const AXIS_END_POINT_Y = LINE_CHART_AXIS_PADDING;
    const Y_TICK_COUNT = 10;
    const Y_TICK_LENGTH = 5;
    const X_TICK_LENGTH = 5;
    const POINT_RADIUS = 2;
    const LINE_PADDING = 0.2;
    const X_GAP = (AXIS_END_POINT_X - AXIS_START_POINT_X) / (lineChartValues.length - 1);
    const BOX_RADIUS = 35;
    const POINT_HOVER_HITBOX = 20;
    const SPACING = 5;
    const NEWLINE_SPACING = 24;
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
        drawLineChart(canvasId, lineChartValues);

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
        console.log("hello");
        drawLineChart(canvasId, lineChartValues);
    };

    /*
    Find the maximum value in the list,
    whether or not the data is integer data or
    real-valued and the number of decimal
    places that values on the x-axis should
    be rounded to.
    */
    for (value of lineChartValues) {
        if (value.y > maxVal) {
            maxVal = value.y;
        }

        intValues &&= Number.isInteger(value.y);

        if (last != undefined && last != value.x) {
            while (last.toFixed(roundDp) == value.x.toFixed(roundDp)) {
                roundDp += 1;
            }
        }

        last = value.x;
    }

    maxVal += (intValues) ? 1 : maxVal * LINE_PADDING;
    minVal = -maxVal;

    LINE_CHART_CTX.strokeStyle = LINE_CHART_STROKE_COLOR;
    LINE_CHART_CTX.beginPath();

    // X Axis
    LINE_CHART_CTX.moveTo(AXIS_START_POINT_X, AXIS_START_POINT_Y);
    LINE_CHART_CTX.lineTo(AXIS_END_POINT_X, AXIS_START_POINT_Y);

    // Y Axis
    LINE_CHART_CTX.moveTo(LINE_CHART_AXIS_PADDING, AXIS_START_POINT_Y);
    LINE_CHART_CTX.lineTo(LINE_CHART_AXIS_PADDING, AXIS_END_POINT_Y);

    // Y Ticks
    var yTickPos = AXIS_START_POINT_Y;

    const step = (intValues) ? 1 : maxVal / Y_TICK_COUNT;
    const stepDp = step.toString().length - 1 /* '.' char */ - Math.round(step).toString().length;

    LINE_CHART_CTX.font = "15px DidactGothic";

    // Y Tick Values
    for (i = minVal; i <= maxVal; i += step) {
        const str = (stepDp <= 0) ? i.toString() : i.toFixed(stepDp).toString();
        const measurements = LINE_CHART_CTX.measureText(str);

        LINE_CHART_CTX.moveTo(AXIS_START_POINT_X, yTickPos);
        LINE_CHART_CTX.lineTo(AXIS_START_POINT_X - Y_TICK_LENGTH, yTickPos);

        LINE_CHART_CTX.fillStyle = LINE_CHART_STROKE_COLOR;
        LINE_CHART_CTX.fillText(str, AXIS_START_POINT_X - Y_TICK_LENGTH - measurements.width,
            yTickPos + (measurements.actualBoundingBoxAscent + measurements.actualBoundingBoxDescent) / 2);

        yTickPos -= (AXIS_START_POINT_Y - AXIS_END_POINT_Y) / ((maxVal * 2) / step);
    }

    // Gap between each point on the x axis
    var previousX;
    var previousY;
    var nextX;

    // Draw the points and lines themselves
    for (i = 0; i < lineChartValues.length; i++) {
        const ENTRY = lineChartValues[i];
        const NAME = ENTRY.x;
        const VALUE = ENTRY.y;
        const X = LINE_CHART_AXIS_PADDING + X_GAP * i;
        const Y = AXIS_END_POINT_Y + (1 - (VALUE / maxVal)) * ((AXIS_START_POINT_Y - AXIS_END_POINT_Y) / 2)
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

        LINE_CHART_CTX.fillText(NAME.toFixed(roundDp), X, AXIS_START_POINT_Y + X_TICK_LENGTH + TEXT_HEIGHT);
        nextX = X + MEASUREMENTS.width;
    }

    LINE_CHART_CTX.stroke();
}