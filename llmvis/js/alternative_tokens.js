/**
 * Draws the alternative tokens visualization.
 * @param {string} canvasId The ID of the canvas that this
 *      alternative tokens visualization should be drawn
 *      to.
 * @param {Object} candidateTokenGroups A list where each
 *      element is another list of token objects representing
 *      the available tokens at each stage.
 * @param {Object} selectedIndices A list containing integers
 *      representing the index of the token that was selected
 *      at each stage, starting at 1.
 * @param {Object} fallbackTokens A list containing additional
 *      token objects that are not any of the candidate tokens.
 *      Should be ordered such that the first element will be
 *      used for the first time the selected index is outside of
 *      the candidate token range, the second element will be
 *      used for the second, and so on.
 */
function drawAlternativeTokens(canvasId, candidateTokenGroups, selectedIndices, fallbackTokens) {
    const CANVAS = document.getElementById(canvasId);
    const CTX = CANVAS.getContext('2d');

    // Redraw the visualization and refit it to the screen.
    const UPDATE = function() {
        const FURTHEST_EXTENT = updateAlternativeTokens(CTX,
            candidateTokenGroups, selectedIndices, fallbackTokens);

        if (FURTHEST_EXTENT > window.innerWidth) {
            CANVAS.width = FURTHEST_EXTENT;
            updateAlternativeTokens(CTX, candidateTokenGroups, selectedIndices, fallbackTokens);
            CANVAS.parentElement.style.width = window.innerWidth.toString() + "px";
        }
    };

    window.addEventListener("resize", UPDATE);
    UPDATE();
}

/**
 * Update the visualization. Called the first time that this visualization
 * is drawn as well as when the screen is resized.
 * @param {Object} ctx The 2D context that should be used for drawing
 *      the visualization.
 * @param {Object} candidateTokenGroups See {@link drawAlternativeTokens}
 * @param {*} selectedIndices See {@link drawAlternativeTokens}
 * @param {*} fallbackTokens See {@link drawAlternativeTokens}
 * @returns A number containing the "furthest extent" of the visualization,
 * i.e. the furthest point drawn as part of the visualization. This can be used
 * to adjust parameters to fit thte visualization on the screen accordingly.
 */
function updateAlternativeTokens(ctx, candidateTokenGroups, selectedIndices, fallbackTokens) {
    const FALLBACK_STACK = fallbackTokens.slice().reverse(); // Slice to create copy
    const STROKE_COLOR = 'rgb(222, 222, 222)';
    const UNSELECTED_TOKEN_COLOR = 'rgb(125, 125, 122)';
    const PADDING = 100;
    const STARTING_X_POSITION = PADDING;
    const STARTING_Y_POSITION = PADDING;
    const X_SPACING = 128;
    const Y_SPACING = 20;
    const FONT_SIZE = 35;
    const CONNECTOR_SPACING = 12;

    var yPosition;
    var xPosition = STARTING_X_POSITION;

    ctx.font = `${FONT_SIZE}px DidactGothic`;
    ctx.strokeStyle = STROKE_COLOR;
    ctx.beginPath();

    /*
    Will be used to store the (x, y) position for
    the end of the most recently drawn selected token,
    so that a connector can be drawn from this point.
    */
    var lastCurvePoint;
    var furthestExtent;

    for (i = 0; i < candidateTokenGroups.length; i++) {
        const GROUP = candidateTokenGroups[i];
        const SELECTED_INDEX = selectedIndices[i];

        yPosition = STARTING_Y_POSITION;

        var maxWidth = -1;
        var lastMaxWidth;
        var lastXPosition;
        var connectorStartX;
        var connectorEndX;
        var foundSelected = false;
        var connectorY;

        for (j = 0; j < GROUP.length; j++) {
            const TOKEN = GROUP[j];
            const TEXT = TOKEN.text;
            const PROB = TOKEN.prob;
            const MEASUREMENTS = ctx.measureText(TEXT);
            const TEXT_WIDTH = MEASUREMENTS.width;
            const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent +
                MEASUREMENTS.actualBoundingBoxDescent;

            if (j == SELECTED_INDEX - 1) {
                /*
                This is not the first group of tokens drawn,
                so we need to draw a connector.
                */
                ctx.fillStyle = STROKE_COLOR;
                connectorStartX = xPosition;
                connectorEndX = xPosition + TEXT_WIDTH;
                connectorY = yPosition - TEXT_HEIGHT/2;
                foundSelected = true;
            } else {
                ctx.fillStyle = UNSELECTED_TOKEN_COLOR;
            }

            ctx.fillText(TEXT, xPosition, yPosition);
        
            yPosition += TEXT_HEIGHT + Y_SPACING;

            if (TEXT_WIDTH > maxWidth) {
                maxWidth = TEXT_WIDTH;
            }
        }

        ctx.fillStyle = UNSELECTED_TOKEN_COLOR;
        ctx.fillText("...", xPosition, yPosition);

        // Selected token might potentially be outside the
        // first n most probable tokens (contained in
        // candidateTokenGroups).
        if (!foundSelected) {
            const FALLBACK_TOKEN = FALLBACK_STACK.pop();
            const MEASUREMENTS = ctx.measureText(FALLBACK_TOKEN.text);
            const TEXT_WIDTH = MEASUREMENTS.width;
            const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent +
                MEASUREMENTS.actualBoundingBoxDescent;

            yPosition += TEXT_HEIGHT + Y_SPACING;
            ctx.fillStyle = STROKE_COLOR;
            ctx.fillText(FALLBACK_TOKEN.text, xPosition, yPosition);
            connectorStartX = xPosition;
            connectorEndX = xPosition + TEXT_WIDTH;
            connectorY = yPosition - TEXT_HEIGHT/2;
        }

        // Remember to draw a connector
        if (lastCurvePoint) {
            ctx.moveTo(lastCurvePoint.x + CONNECTOR_SPACING, lastCurvePoint.y);
            ctx.lineTo(lastXPosition + lastMaxWidth, lastCurvePoint.y);
            ctx.bezierCurveTo(
                lastXPosition + lastMaxWidth + X_SPACING/2, lastCurvePoint.y,
                lastXPosition + lastMaxWidth + X_SPACING/2, connectorY,
                connectorStartX - CONNECTOR_SPACING, connectorY 
            )
        }

        lastCurvePoint = {x: connectorEndX, y: connectorY};
        lastMaxWidth = maxWidth;
        lastXPosition = xPosition;
        furthestExtent = xPosition + maxWidth + PADDING;
        xPosition += maxWidth + X_SPACING;
    }

    ctx.stroke();
    return furthestExtent;
}