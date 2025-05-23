/**
 * Draw a new Radar Chart visualization to a given canvas.
 * @param {string} canvasId The ID of the canvas that this visualization
 *      should be drawn to.
 * @param {Array} values The data that this visualization should show.
 *      Each element of this array should be another array where the
 *      first element is the categorical value and the second is the
 *      associated numerical value.
 */
function drawRadarChart(canvasId, values) {
    const CANVAS = document.getElementById(canvasId);
    const CTX = CANVAS.getContext("2d");
    const STROKE_COLOR = 'rgb(222, 222, 222)';

    var maxVal = undefined;
    var minVal = undefined;

    // Clear canvas in case line charts have been drawn before
    CTX.clearRect(0, 0, CANVAS.width, CANVAS.height);

    // First pass: find maximum value
    for (value of values) {
        if (maxVal == undefined || value[1] > maxVal) {
            maxVal = value[1];
        }

        if (minVal == undefined || value[1] < minVal) {
            minVal = value[1];
        }
    }

    // Background polygon
    drawPolygon(CTX, CANVAS, Array(values.length).fill(["", 1]), STROKE_COLOR, 1, 0, false);
    // Data polygon
    drawPolygon(CTX, CANVAS, values, 'rgb(117, 115, 138)', maxVal, minVal, true);

    enableResizing(CANVAS, function() {
        drawRadarChart(canvasId, values);
    });
}

/**
 * Draw a polygon to a canvas.
 * @param {Object} ctx The 2D context that this polygon should
 *      be drawn with.
 * @param {Object} canvas The canvas that this polygon should
 *      be drawn to.
 * @param {Array} values The data that this polygon should show.
 * @param {string} color The color of this polygon.
 * @param {number} maxVal The maximum value that any point in
 *      this polygon will contain.
 * @param {boolean} fill Set to `true` to fill this polygon or
 *      `false` to not fill it.
 */
function drawPolygon(ctx, canvas, values, color, maxVal, minVal, fill) {
    const PADDING = 60;
    // Use the smaller of the width or the height to ensure that it
    // remains onscreen.
    const RADIUS = Math.min(canvas.width, canvas.height)/2 - PADDING;
    const LABEL_DISTANCE = 8;
    const ROTATE_ANGLE = (2*Math.PI)/values.length;

    // Vector for where to draw each point around the circle
    // Start at the "top" of the circle
    var v = [0, -RADIUS];

    var firstMove = true;
    var firstX;
    var firstY;

    ctx.strokeStyle = color;
    ctx.font = "15px DidactGothic";

    var textDrawPositions = [];

    ctx.beginPath();

    for (value of values) {
        const NAME = value[0];
        const PROPORTION = value[1];
        const LOCAL_X = v[0];
        const LOCAL_Y = v[1];
        const MULTIPLIER = (PROPORTION-minVal) / (maxVal-minVal);
        const GLOBAL_X = canvas.width/2 + LOCAL_X*MULTIPLIER;
        const GLOBAL_Y = canvas.height/2 + LOCAL_Y*MULTIPLIER;

        if (firstMove) {
            ctx.moveTo(GLOBAL_X, GLOBAL_Y);
            firstMove = false;
            firstX = GLOBAL_X;
            firstY = GLOBAL_Y;
        } else {
            ctx.lineTo(GLOBAL_X, GLOBAL_Y);
        }

        // Text
        const MEASUREMENTS = ctx.measureText(NAME);
        const TEXT_WIDTH = MEASUREMENTS.width;
        const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent +
            MEASUREMENTS.actualBoundingBoxDescent;
        const VECT_MAGNITUDE = Math.sqrt((LOCAL_X*LOCAL_X) + (LOCAL_Y*LOCAL_Y));
        const NORMALIZED_X = LOCAL_X/VECT_MAGNITUDE;
        const NORMALIZED_Y = LOCAL_Y/VECT_MAGNITUDE;

        var textX = canvas.width/2 + LOCAL_X - TEXT_WIDTH/2 + (TEXT_WIDTH/2 + LABEL_DISTANCE)*NORMALIZED_X;
        var textY = canvas.height/2 + LOCAL_Y + TEXT_HEIGHT/2 + (TEXT_HEIGHT/2 + LABEL_DISTANCE)*NORMALIZED_Y;

        // Clamp to screen
        if (textX > canvas.width - TEXT_WIDTH) {
            textX = canvas.width - TEXT_WIDTH;
        }

        if (textX < 0) {
            textX = 0;
        }

        if (textY > canvas.height - TEXT_HEIGHT) {
            textY = canvas.height - TEXT_HEIGHT;
        }

        if (textY < 0) {
            textY = 0;
        }

        textDrawPositions.push([NAME, textX, textY]);

        // Rotate
        v = [
            LOCAL_X*Math.cos(ROTATE_ANGLE) - LOCAL_Y*Math.sin(ROTATE_ANGLE),
            LOCAL_X*Math.sin(ROTATE_ANGLE) + LOCAL_Y*Math.cos(ROTATE_ANGLE)
        ];
    }
    
    ctx.lineTo(firstX, firstY);
    ctx.fillStyle = color;

    if (fill) {
        ctx.fill();
    }
    
    ctx.stroke();

    ctx.fillStyle = 'white';
    for (position of textDrawPositions) {
        ctx.fillText(position[0],
            position[1],
            position[2]);
    }
}