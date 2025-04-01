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
    }

    const FULL_BAR_WIDTH = (AXIS_END_POINT_X - AXIS_START_POINT_X) / barChartValues.length;
    const BAR_CHART_GAP = FULL_BAR_WIDTH / 6;
    const BAR_WIDTH = FULL_BAR_WIDTH - (BAR_CHART_GAP*2);
    const Y_TICK_COUNT = 10;
    const Y_TICK_LENGTH = 5;
    const X_TICK_LENGTH = 5;
    const X_AXIS_TICKS_ROTATION_DEGREE = 45;
    const X_AXIS_TICKS_MARGIN = 10;
    const X_AXIS_LABEL_SPACING = 4;
    const Y_AXIS_LABEL_SPACING = 10;

    drawAxis(
        BAR_CHART_CTX,
        BAR_CHART_AXIS_PADDING,
        BAR_CHART_AXIS_PADDING,
        BAR_CHART_STROKE_COLOR,
        categoricalData(barChartValues.map((x) => x[0])),
        AxisPosition.BOTTOM
    );

    const AXIS_DRAW_RESULTS = drawAxis(
        BAR_CHART_CTX,
        BAR_CHART_AXIS_PADDING,
        BAR_CHART_AXIS_PADDING,
        BAR_CHART_STROKE_COLOR,
        continuousData(0, maxVal, Math.round(maxVal)),
        AxisPosition.LEFT
    )

    maxVal = AXIS_DRAW_RESULTS.max;
    var x = AXIS_START_POINT_X + BAR_CHART_GAP;

    for (var i = 0; i < barChartValues.length; i++) {
        const ENTRY = barChartValues[i];
        const VALUE = ENTRY[1];

        BAR_CHART_CTX.fillStyle = calculateRgb(i, barChartValues.length - 1, 0);
        BAR_CHART_CTX.fillRect(x, AXIS_START_POINT_Y, BAR_WIDTH, -((VALUE / maxVal) * (AXIS_START_POINT_Y - AXIS_END_POINT_Y)));

        x += BAR_WIDTH + BAR_CHART_GAP*2;
    }

    enableResizing(BAR_CHART_CANVAS, function() {
        drawBarChart(canvasId, barChartValues, xLabel, yLabel)
    });
}