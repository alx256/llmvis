/**
 * Draw an AI classifier visualization
 * 
 * @param {string} canvasId The ID of the canvas that this
 *      visualization should be drawn to.
 * @param {Object} classifiedData The data that has been
 *      classified into classes that should be shown by
 *      this visualization. Each element of this list should
 *      consist of two elements: the first with the name of the
 *      class and the second containg a list with all the points
 *      that have been classified into that class.
 * @param {Object} points A list containing all the available
 *      points that this classifier visualization could have.
 */
function drawAiClassifier(canvasId, classifiedData, points, xLabel, yLabel) {
    const CLASSIFIER_CANVAS = document.getElementById(canvasId);
    const CLASSIFIER_CTX = CLASSIFIER_CANVAS.getContext('2d');
    const LABEL_MARGINS = 8;

    CLASSIFIER_CTX.font = "20px DidactGothic";

    var classTextMeasurements = [];
    var axisLabelHeight;
    var maxClassTextMeasurement = -1
    var maxVal = -1;
    var minVal = -1;
    var roundDp = 0;
    var last = undefined;

    /*
    Initial pass: find the maximum and minimum points as well as the
    number of decimal points that the ticks on the x axis should be
    rounded to.
    */
    for (point of points) {
        if (point > maxVal) {
            maxVal = point;
        }
    
        if (minVal == -1 || point < minVal) {
            minVal = point;
        }

        if (last != undefined && last != point) {
            // Find the number of decimal points to round the points to
            // where they can still be differentiable from one another.
            while (last.toFixed(roundDp) == point.toFixed(roundDp)) {
                roundDp += 1;
            }
        }

        last = point;
    }

    // Second pass: determine the padding that the y-axis
    // has to fit in as much text as possible.
    for (data of classifiedData) {
        const MEASUREMENT = CLASSIFIER_CTX.measureText(data[0]);
        const WIDTH = MEASUREMENT.width + LABEL_MARGINS*2;
        const HEIGHT = MEASUREMENT.actualBoundingBoxAscent +
            MEASUREMENT.actualBoundingBoxDescent + LABEL_MARGINS*2;

        classTextMeasurements.push(WIDTH);

        if (WIDTH > maxClassTextMeasurement) {
            maxClassTextMeasurement = WIDTH;
        }

        if (axisLabelHeight == undefined) {
            axisLabelHeight = HEIGHT;
        }
    }

    const STROKE_COLOR = 'rgb(222, 222, 222)';
    const AXIS_PADDING_X = maxClassTextMeasurement;
    const AXIS_PADDING_Y = 56;

    const AXIS_START_POINT_X = AXIS_PADDING_X;
    const AXIS_START_POINT_Y = CLASSIFIER_CANVAS.height - AXIS_PADDING_Y;
    const AXIS_END_POINT_X = CLASSIFIER_CANVAS.width - AXIS_PADDING_X;
    const AXIS_END_POINT_Y = AXIS_PADDING_Y;
    const RECT_RADIUS = 17;
    const RECT_MARGINS = 10;

    CLASSIFIER_CTX.clearRect(0, 0, CLASSIFIER_CANVAS.width, CLASSIFIER_CANVAS.height);

    // X Axis
    drawAxis(CLASSIFIER_CTX,
        maxClassTextMeasurement,
        AXIS_PADDING_Y,
        STROKE_COLOR,
        categoricalData(points),
        AxisPosition.BOTTOM,
        xLabel
    )

    // Y Axis
    drawAxis(CLASSIFIER_CTX,
        maxClassTextMeasurement,
        AXIS_PADDING_Y,
        STROKE_COLOR,
        categoricalData(classifiedData.map((d) => d[0])),
        AxisPosition.LEFT,
        yLabel
    )

    const Y_AXIS_SEGMENT_HEIGHT = (AXIS_START_POINT_Y - AXIS_END_POINT_Y) / classifiedData.length;
    const X_AXIS_SEGMENT_WIDTH = (AXIS_END_POINT_X - AXIS_START_POINT_X) / points.length;
    var yPosition = AXIS_END_POINT_Y;

    /*
    Draw point labels on the x-axis.
    This can be done in the main loop that draws the rects, but
    the problem is that if a point is not put into any class
    (which can happen) then it will not be drawn. To be safe,
    we make sure that all are drawn in this loop.
    */
    var nextX;

    for (var i = 0; i < points.length; i++) {
        const X = AXIS_START_POINT_X + (i/(points.length-1))*(AXIS_END_POINT_X - AXIS_START_POINT_X);
        const POINT = points[i];

        if (nextX != undefined && X < nextX) {
            continue;
        }

        const STR = POINT.toFixed(roundDp);
        const MEASUREMENTS = CLASSIFIER_CTX.measureText(STR);

        CLASSIFIER_CTX.fillStyle = STROKE_COLOR;
        nextX = X + MEASUREMENTS.width;
    }

    /*
    Main loop. Draw all the rects.
    */
    for (var i = 0; i < classifiedData.length; i++) {
        const CLASS = classifiedData[i][0];
        const MATCHES = classifiedData[i][1];
        const MEASURED_WIDTH = classTextMeasurements[i];
        const RECT_COLOR = calculateRgb(i, classifiedData.length - 1, 0);

        CLASSIFIER_CTX.fillStyle = STROKE_COLOR;

        var lastX = undefined;
        var lastIndex = undefined;
        // Number of connected rects
        var connectors = 1;

        for (var match of MATCHES) {
            const INDEX = points.indexOf(match);

            if (INDEX == -1) {
                // Misclassification
                continue;
            }

            const X = AXIS_START_POINT_X + (INDEX/points.length)*(AXIS_END_POINT_X - AXIS_START_POINT_X);
            // Determine if this rect should be connected to the previous one
            // Occurs if the distance between this point and the previous point
            // is roughly the expected distance between two points and hence the two
            // rects are back-to-back. Because we are dealing with floating-point
            // arithmetic there is a margin of error, so check if the difference is
            // less than some small value.
            const SHOULD_CONNECT = lastIndex != undefined && INDEX == lastIndex + 1;

            if (SHOULD_CONNECT) {
                connectors += 1;
            }
            
            const RECT_X = (SHOULD_CONNECT) ? lastX : X + RECT_MARGINS;
            const RECT_Y = yPosition;
            const RECT_WIDTH = X_AXIS_SEGMENT_WIDTH*connectors;
            const RECT_HEIGHT = Y_AXIS_SEGMENT_HEIGHT - RECT_MARGINS;

            CLASSIFIER_CTX.beginPath();
            CLASSIFIER_CTX.strokeStyle = RECT_COLOR;
            CLASSIFIER_CTX.fillStyle = RECT_COLOR;
            CLASSIFIER_CTX.roundRect(RECT_X, RECT_Y, RECT_WIDTH, RECT_HEIGHT, RECT_RADIUS);
            CLASSIFIER_CTX.fill();
            CLASSIFIER_CTX.stroke();
            CLASSIFIER_CTX.strokeStyle = STROKE_COLOR;
            CLASSIFIER_CTX.fillStyle = STROKE_COLOR;

            lastIndex = INDEX;

            if (!SHOULD_CONNECT) {
                lastX = RECT_X;
                connectors = 1;
            }
        }

        yPosition += Y_AXIS_SEGMENT_HEIGHT;
    }

    enableResizing(CLASSIFIER_CANVAS, function() {
        drawAiClassifier(canvasId, classifiedData, points, xLabel, yLabel);
    });
}