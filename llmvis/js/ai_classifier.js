var classifiedData;
var points;

/**
 * Draw an AI classifier visualization
 */
function drawAiClassifier() {
    const CLASSIFIER_CANVAS = document.getElementById('llmvis-ai-classifier-canvas');
    const CLASSIFIER_CTX = CLASSIFIER_CANVAS.getContext('2d');
    const LABEL_MARGINS = 8;

    CLASSIFIER_CTX.font = "20px DidactGothic";

    var classTextMeasurements = [];
    var axisLabelHeight;
    var maxClassTextMeasurement = -1
    var maxVal = -1;
    var minVal = -1;

    // Initial pass: find the maximum and minimum points
    for (point of points) {
        if (point > maxVal) {
            maxVal = point;
        }
    
        if (minVal == -1 || point < minVal) {
            minVal = point;
        }
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

    CLASSIFIER_CTX.strokeStyle = STROKE_COLOR;
    CLASSIFIER_CTX.beginPath();

    // X Axis
    CLASSIFIER_CTX.moveTo(AXIS_START_POINT_X, AXIS_START_POINT_Y);
    CLASSIFIER_CTX.lineTo(AXIS_END_POINT_X, AXIS_START_POINT_Y);

    // Y Axis
    CLASSIFIER_CTX.moveTo(AXIS_PADDING_X, AXIS_START_POINT_Y);
    CLASSIFIER_CTX.lineTo(AXIS_PADDING_X, AXIS_END_POINT_Y);

    CLASSIFIER_CTX.stroke();

    const Y_AXIS_SEGMENT_HEIGHT = (AXIS_START_POINT_Y - AXIS_END_POINT_Y) / classifiedData.length;
    const X_AXIS_SEGMENT_WIDTH = (AXIS_END_POINT_X - AXIS_START_POINT_X) / points.length;
    var yPosition = AXIS_END_POINT_Y +  Y_AXIS_SEGMENT_HEIGHT/2;

    const MAX_COLOR_VALUE = 158;

    var rgb = [MAX_COLOR_VALUE, 0, 0];
    var channel = 1;
    var multiplier = 1;

    const INCREMENT = MAX_COLOR_VALUE/classifiedData.length * 3;

    /*
    Draw point labels on the x-axis.
    This can be done in the main loop that draws the rects, but
    the problem is that if a point is not put into any class
    (which can happen) then it will not be drawn. To be safe,
    we make sure that all are drawn in this loop.
    */
    for (point of points) {
        const NORMALIZED = normalized(point, maxVal, minVal, points.length);
        const X = AXIS_START_POINT_X + NORMALIZED*(AXIS_END_POINT_X - AXIS_START_POINT_X);
        CLASSIFIER_CTX.fillStyle = STROKE_COLOR;
        CLASSIFIER_CTX.fillText(point, X, AXIS_START_POINT_Y + axisLabelHeight + LABEL_MARGINS);
    }

    /*
    Main loop. Draw all the rects.
    */
    for (i = 0; i < classifiedData.length; i++) {
        const CLASS = classifiedData[i][0];
        const MATCHES = classifiedData[i][1];
        const MEASURED_WIDTH = classTextMeasurements[i];
        const RECT_COLOR = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

        CLASSIFIER_CTX.fillStyle = STROKE_COLOR;
        CLASSIFIER_CTX.fillText(CLASS, AXIS_START_POINT_X - MEASURED_WIDTH + LABEL_MARGINS, yPosition);

        var last = undefined;
        var lastX = undefined;
        // Number of connected rects
        var connectors = 1;

        for (match of MATCHES) {
            const NORMALIZED = normalized(match, maxVal, minVal, points.length);
            const X = AXIS_START_POINT_X + NORMALIZED*(AXIS_END_POINT_X - AXIS_START_POINT_X);
            // Determine if this rect should be connected to the previous one
            // Occurs if the distance between this point and the previous point
            // is roughly the expected distance between two points and hence the two
            // rects are back-to-back. Because we are dealing with floating-point
            // arithmetic there is a margin of error, so check if the difference is
            // less than some small value.
            const SHOULD_CONNECT = last != undefined && Math.abs(last - match) - (maxVal-minVal)/(points.length-1) < 0.01;
            
            if (SHOULD_CONNECT) {
                connectors += 1;
            }
            
            const RECT_X = (SHOULD_CONNECT) ? lastX : X + RECT_MARGINS;
            const RECT_Y = yPosition - Y_AXIS_SEGMENT_HEIGHT/2 + RECT_MARGINS;
            const RECT_WIDTH = X_AXIS_SEGMENT_WIDTH*connectors - RECT_MARGINS;
            const RECT_HEIGHT = Y_AXIS_SEGMENT_HEIGHT - RECT_MARGINS;

            CLASSIFIER_CTX.beginPath();
            CLASSIFIER_CTX.strokeStyle = RECT_COLOR;
            CLASSIFIER_CTX.fillStyle = RECT_COLOR;
            CLASSIFIER_CTX.roundRect(RECT_X, RECT_Y, RECT_WIDTH, RECT_HEIGHT, RECT_RADIUS);
            CLASSIFIER_CTX.fill();
            CLASSIFIER_CTX.stroke();
            CLASSIFIER_CTX.strokeStyle = STROKE_COLOR;
            CLASSIFIER_CTX.fillStyle = STROKE_COLOR;

            last = match;

            if (!SHOULD_CONNECT) {
                lastX = RECT_X;
                connectors = 1;
            }
        }

        // Ensure each class has a different color
        rgb[channel] += INCREMENT*multiplier;

        if (rgb[channel] >= MAX_COLOR_VALUE) {
            rgb[channel] = MAX_COLOR_VALUE;
            multiplier *= -1;
            channel = (channel - 1) % 3;
        }
        
        yPosition += Y_AXIS_SEGMENT_HEIGHT;
    }
}

/**
 * Normalize a point. This will result in a value between `0.0` and
 * `1.0` where `0.0` is the smallest point and `1.0` is the last point +
 * the distance between any two points. Note that `1.0` does not correlate
 * to the maximum point value because we want the last point to be drawn
 * slightly inwards on the x-axis.
 * @param {*} point The point value that should be normalized
 * @param {*} maxVal The maximum point value
 * @param {*} minVal The minimum point value
 * @param {*} pointsCount The number of points
 * @returns A value from `0.0` to `1.0` representing a normalized point.
 */
function normalized(point, maxVal, minVal, pointsCount) {
    return (point - minVal)/((maxVal + (maxVal - minVal)/(pointsCount-1)) - minVal);
}