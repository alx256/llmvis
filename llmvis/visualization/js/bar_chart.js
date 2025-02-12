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
 */
function drawBarChart(canvasId, barChartValues) {
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

    BAR_CHART_CTX.strokeStyle = BAR_CHART_STROKE_COLOR;
    BAR_CHART_CTX.beginPath();
    
    // X Axis
    BAR_CHART_CTX.moveTo(AXIS_START_POINT_X, AXIS_START_POINT_Y);
    BAR_CHART_CTX.lineTo(AXIS_END_POINT_X, AXIS_START_POINT_Y);

    // Y Axis
    BAR_CHART_CTX.moveTo(BAR_CHART_AXIS_PADDING, AXIS_START_POINT_Y);
    BAR_CHART_CTX.lineTo(BAR_CHART_AXIS_PADDING, AXIS_END_POINT_Y);
    
    // Y Ticks
    var yTickPos = AXIS_START_POINT_Y;
    var maxBarHeight = -1;

    // TODO: Increase step for integer values if max value is too high
    const step = (intValues) ? 1 : maxVal / Y_TICK_COUNT;
    const stepDp = step.toString().length - 1 /* '.' char */ - Math.round(step).toString().length;
    
    BAR_CHART_CTX.font = "15px DidactGothic";

    // Y Tick Values
    for (i = 0; i <= maxVal; i += step) {
        const str = (stepDp <= 0) ? i.toString() : i.toFixed(stepDp).toString();
        const measurements = BAR_CHART_CTX.measureText(str);

        BAR_CHART_CTX.moveTo(AXIS_START_POINT_X, yTickPos);
        BAR_CHART_CTX.lineTo(AXIS_START_POINT_X - Y_TICK_LENGTH, yTickPos);
    
        BAR_CHART_CTX.fillStyle = BAR_CHART_STROKE_COLOR;
        BAR_CHART_CTX.fillText(str, AXIS_START_POINT_X - Y_TICK_LENGTH - measurements.width,
            yTickPos + (measurements.actualBoundingBoxAscent + measurements.actualBoundingBoxDescent) / 2);

        if (i == maxVal) {
            // Store the y tick position for the highest value.
            // This is used for drawing the bars as a fraction of this
            maxBarHeight = yTickPos;
        }

        yTickPos -= (AXIS_START_POINT_Y - AXIS_END_POINT_Y) / (maxVal / step);
    }

    // Used for assigning a different colour to each bar
    var rgb = [MAX_COLOR_VALUE, 0, 0];
    var channel = 1;
    var multiplier = 1;

    const INCREMENT = MAX_COLOR_VALUE / barChartValues.length * 3;

    for (i = 0; i < barChartValues.length; i++) {
        const ENTRY = barChartValues[i];
        const NAME = ENTRY[0];
        const VALUE = ENTRY[1];
        const X = BAR_CHART_AXIS_PADDING + BAR_CHART_GAP + (BAR_WIDTH + BAR_CHART_GAP)*i;
        const TICK_X = X + BAR_WIDTH / 2;
        const MEASUREMENTS = BAR_CHART_CTX.measureText(NAME);
        const TEXT_WIDTH = MEASUREMENTS.width;
        const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent + MEASUREMENTS.actualBoundingBoxDescent;

        BAR_CHART_CTX.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
        BAR_CHART_CTX.fillRect(X, AXIS_START_POINT_Y, BAR_WIDTH, -((VALUE / maxVal) * (AXIS_START_POINT_Y - AXIS_END_POINT_Y)));

        rgb[channel] += INCREMENT*multiplier;

        if (rgb[channel] >= MAX_COLOR_VALUE) {
            rgb[channel] = MAX_COLOR_VALUE;
            multiplier *= -1;
            channel = (channel - 1) % 3;
        }

        // X-Axis Categorical Labels
        BAR_CHART_CTX.save();
        BAR_CHART_CTX.translate(TICK_X, AXIS_START_POINT_Y + X_TICK_LENGTH);
        BAR_CHART_CTX.rotate(Math.PI / 2.5);
        BAR_CHART_CTX.fillStyle = BAR_CHART_STROKE_COLOR;
        BAR_CHART_CTX.fillText(NAME, 0, 0);
        BAR_CHART_CTX.restore();
    }

    BAR_CHART_CTX.stroke();
}