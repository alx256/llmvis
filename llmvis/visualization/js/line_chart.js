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

    var roundDp = 0;
    var last;

    /*
    Find the maximum value in the list,
    whether or not the data is integer data or
    real-valued and the number of decimal
    places that values on the x-axis should
    be rounded to.
    */
    for (value of lineChartValues) {
        if (value[1] > maxVal) {
            maxVal = value[1];
        }

        intValues &&= Number.isInteger(value[1]);

        if (last != undefined && last != value[0]) {
            while (last.toFixed(roundDp) == value[0].toFixed(roundDp)) {
                roundDp += 1;
            }
        }

        last = value[0];
    }

    maxVal += (intValues) ? 1 : maxVal*LINE_PADDING;
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

        yTickPos -= (AXIS_START_POINT_Y - AXIS_END_POINT_Y) / ((maxVal*2) / step);
    }

    // Gap between each point on the x axis
    var xGap = (AXIS_END_POINT_X - AXIS_START_POINT_X) / (lineChartValues.length - 1);
    var previousX;
    var previousY;
    var nextX;

    // Draw the points and lines themselves
    for (i = 0; i < lineChartValues.length; i++) {
        const ENTRY = lineChartValues[i];
        const NAME = ENTRY[0];
        const VALUE = ENTRY[1];
        const X = LINE_CHART_AXIS_PADDING + xGap*i;
        const Y = AXIS_END_POINT_Y + (1 - (VALUE / maxVal))*((AXIS_START_POINT_Y - AXIS_END_POINT_Y)/2)
        const MEASUREMENTS = LINE_CHART_CTX.measureText(NAME.toFixed(roundDp));
        const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent + MEASUREMENTS.actualBoundingBoxDescent;

        // Line from previous point (if applicable)
        if (previousX != undefined && previousY != undefined) {
            LINE_CHART_CTX.moveTo(previousX, previousY);
            LINE_CHART_CTX.lineTo(X, Y);
        }

        LINE_CHART_CTX.moveTo(X + 2, Y);
        // Point (as a filled circle)
        LINE_CHART_CTX.arc(X, Y, POINT_RADIUS, 0, 2*Math.PI);
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