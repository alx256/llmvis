/**
 * Draw a bar chart visualization
 * 
 * @param {string} canvasId The ID of the canvas that this
 *      bar chart should be drawn to.
 * @param {Object} barChartValues The values that this bar
 *      chart should visualize. Should be a list where each
 *      element is another list where the first element is
 *      the categorical value and the second element is the
 *      numerical value.
 * @param {string} xLabel The label that should be shown under
 *      the x-axis to describe it or an empty string to show no
 *      label.
 * @param {string} yLabel The label that should be shown next to
 *      the y-axis to describe it or an empty string to show no
 *      label.
 */
function drawBarChart(canvasId, barChartValues, xLabel, yLabel) {
    const BAR_CHART_CANVAS = document.getElementById(canvasId);
    const BAR_CHART_CTX = BAR_CHART_CANVAS.getContext('2d');

    const BAR_CHART_STROKE_COLOR = 'rgb(222, 222, 222)';
    const BAR_CHART_AXIS_PADDING = 56;
    const AXIS_START_POINT_X = BAR_CHART_AXIS_PADDING;
    const AXIS_START_POINT_Y = BAR_CHART_CANVAS.height - BAR_CHART_AXIS_PADDING;
    const AXIS_END_POINT_X = BAR_CHART_CANVAS.width - BAR_CHART_AXIS_PADDING;
    const AXIS_END_POINT_Y = BAR_CHART_AXIS_PADDING;

    var maxVal = -1;
    var intValues = true;

    /*
    Initial pass through the values.
    Find the maximimum value so that we
    can scale the axes accordingly as well
    as determine if all values are integers
    or if we are dealing with real values.
    */
    for (value of barChartValues) {
        if (value[1] > maxVal) {
            maxVal = value[1];
        }

        intValues &&= Number.isInteger(value[1]);
    }

    const FULL_BAR_WIDTH = (AXIS_END_POINT_X - AXIS_START_POINT_X) / barChartValues.length;
    const BAR_CHART_GAP = FULL_BAR_WIDTH / 6;
    const BAR_WIDTH = FULL_BAR_WIDTH - (BAR_CHART_GAP*2);
    const Y_TICK_COUNT = 10;
    const Y_TICK_LENGTH = 5;
    const X_TICK_LENGTH = 5;
    const MAX_COLOR_VALUE = 158;
    const X_AXIS_TICKS_ROTATION_DEGREE = 45;
    const X_AXIS_TICKS_MARGIN = 10;
    const X_AXIS_LABEL_SPACING = 4;
    const Y_AXIS_LABEL_SPACING = 10;

    BAR_CHART_CTX.strokeStyle = BAR_CHART_STROKE_COLOR;
    BAR_CHART_CTX.beginPath();
    
    BAR_CHART_CTX.font = "15px DidactGothic";

    // X Axis
    BAR_CHART_CTX.moveTo(AXIS_START_POINT_X, AXIS_START_POINT_Y);
    BAR_CHART_CTX.lineTo(AXIS_END_POINT_X, AXIS_START_POINT_Y);

    // Y Axis
    BAR_CHART_CTX.moveTo(BAR_CHART_AXIS_PADDING, AXIS_START_POINT_Y);
    BAR_CHART_CTX.lineTo(BAR_CHART_AXIS_PADDING, AXIS_END_POINT_Y);
    
    // Y Ticks
    var yTickPos = AXIS_START_POINT_Y;
    var maxBarHeight = -1;

    const step = maxVal / Y_TICK_COUNT;

    // Y Tick Values
    var maxYTick = -1;
    var yPoint = 0;

    for (var i = 0; i <= Y_TICK_COUNT; i++) {
        const str = (Number.isInteger(yPoint)) ? yPoint.toString() : yPoint.toPrecision(2).toString();
        const measurements = BAR_CHART_CTX.measureText(str);

        BAR_CHART_CTX.moveTo(AXIS_START_POINT_X, yTickPos);
        BAR_CHART_CTX.lineTo(AXIS_START_POINT_X - Y_TICK_LENGTH, yTickPos);
    
        BAR_CHART_CTX.fillStyle = BAR_CHART_STROKE_COLOR;
        BAR_CHART_CTX.fillText(str, AXIS_START_POINT_X - Y_TICK_LENGTH - measurements.width,
            yTickPos + (measurements.actualBoundingBoxAscent + measurements.actualBoundingBoxDescent) / 2);

        if (yPoint > maxBarHeight) {
            // Store the y tick position for the highest value.
            // This is used for drawing the bars as a fraction of this
            maxBarHeight = yTickPos;
        }

        yTickPos -= (AXIS_START_POINT_Y - AXIS_END_POINT_Y) / (maxVal / step);
        yPoint += step;

        if (measurements.width > maxYTick) {
            maxYTick = measurements.width;
        }
    }

    var channel = 1;
    var multiplier = 1;

    const INCREMENT = MAX_COLOR_VALUE / barChartValues.length * 3;
    var maxOpposite = -1;

    for (var i = 0; i < barChartValues.length; i++) {
        const ENTRY = barChartValues[i];
        const NAME = ENTRY[0];
        const VALUE = ENTRY[1];
        const X = BAR_CHART_AXIS_PADDING + BAR_CHART_GAP + (BAR_WIDTH + BAR_CHART_GAP)*i;
        const TICK_X = X + BAR_WIDTH / 2;
        const MEASUREMENTS = BAR_CHART_CTX.measureText(NAME);
        const TEXT_WIDTH = MEASUREMENTS.width;
        const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent + MEASUREMENTS.actualBoundingBoxDescent;

        BAR_CHART_CTX.fillStyle = calculateRgb(i, barChartValues.length - 1, 0);
        BAR_CHART_CTX.fillRect(X, AXIS_START_POINT_Y, BAR_WIDTH, -((VALUE / maxVal) * (AXIS_START_POINT_Y - AXIS_END_POINT_Y)));

        // X-Axis Categorical Labels
        BAR_CHART_CTX.save();
        BAR_CHART_CTX.translate(TICK_X, AXIS_START_POINT_Y + X_TICK_LENGTH + X_AXIS_TICKS_MARGIN);
        BAR_CHART_CTX.rotate(X_AXIS_TICKS_ROTATION_DEGREE*Math.PI/180);
        BAR_CHART_CTX.fillStyle = BAR_CHART_STROKE_COLOR;
        BAR_CHART_CTX.fillText(NAME, 0, 0);
        BAR_CHART_CTX.restore();

        /*
        Calculate the opposite side length in the right-angle triangle formed with
        Hypoteneuse = FONT_WIDTH
        Theta = X_AXIS_TICKS_ROTATION_DEGREE

        Calculating this will allow us to find the longest distance between the x axis
        and the end of a label so we can draw the x axis label accordingly.
        */
        const OPPOSITE_SIDE_LENGTH = Math.sin(X_AXIS_TICKS_ROTATION_DEGREE) * TEXT_WIDTH;

        if (OPPOSITE_SIDE_LENGTH > maxOpposite) {
            maxOpposite = OPPOSITE_SIDE_LENGTH;
        }
    }

    // X Axis Label
    const X_LABEL_MEASURMENTS = BAR_CHART_CTX.measureText(xLabel);
    const X_LABEL_HEIGHT = X_LABEL_MEASURMENTS.actualBoundingBoxAscent + X_LABEL_MEASURMENTS.actualBoundingBoxDescent;
    const X_LABEL_X = AXIS_START_POINT_X + (AXIS_END_POINT_X - AXIS_START_POINT_X)/2 - X_LABEL_MEASURMENTS.width/2;
    const X_LABEL_Y = AXIS_START_POINT_Y + X_LABEL_HEIGHT + X_AXIS_TICKS_MARGIN + maxOpposite + X_AXIS_LABEL_SPACING;
    BAR_CHART_CTX.fillStyle = BAR_CHART_STROKE_COLOR;
    // Draw label right on edge of canvas if it is going off the canvas
    BAR_CHART_CTX.fillText(xLabel,
        X_LABEL_X, (X_LABEL_Y > BAR_CHART_CANVAS.height) ? BAR_CHART_CANVAS.height : X_LABEL_Y);

    // Y Axis Label
    const Y_LABEL_MEASUREMENTS = BAR_CHART_CTX.measureText(yLabel);
    const Y_LABEL_HEIGHT = Y_LABEL_MEASUREMENTS.actualBoundingBoxAscent + Y_LABEL_MEASUREMENTS.actualBoundingBoxDescent;
    const Y_LABEL_X = AXIS_START_POINT_X - Y_TICK_LENGTH - maxYTick - Y_AXIS_LABEL_SPACING;
    const Y_LABEL_Y = (AXIS_START_POINT_Y + AXIS_END_POINT_Y) / 2 + Y_LABEL_MEASUREMENTS.width/2;
    BAR_CHART_CTX.save();
    BAR_CHART_CTX.translate((Y_LABEL_X  - Y_LABEL_HEIGHT < 0) ? Y_LABEL_HEIGHT : Y_LABEL_X, Y_LABEL_Y);
    BAR_CHART_CTX.rotate(-90*Math.PI/180);
    BAR_CHART_CTX.fillStyle = BAR_CHART_STROKE_COLOR;
    BAR_CHART_CTX.fillText(yLabel, 0, 0);
    BAR_CHART_CTX.restore();

    BAR_CHART_CTX.stroke();
}