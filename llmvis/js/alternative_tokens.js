/**
 * Draws the alternative tokens visualization.
 * @param {string} canvasId The ID of the canvas that this
 *      alternative tokens visualization should be drawn
 *      to.
 * @param {string} legendId The ID of the canvas that will
 *      contain the legend for this alternative tokens
 *      visualization.
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
function drawAlternativeTokens(canvasId, legendId, candidateTokenGroups, selectedIndices, fallbackTokens) {
    const CANVAS = document.getElementById(canvasId);
    const CTX = CANVAS.getContext('2d');
    const TOOLTIP_WIDTH = 200;
    const TOOLTIP_HEIGHT = 80;
    const SCROLL_SCALE = 0.00055;
    const LEGEND_CANVAS = document.getElementById(legendId);
    const LEGEND_CTX = LEGEND_CANVAS.getContext('2d');
    const PALETTE = [[176, 46, 52], [180, 157, 46], [48, 147, 38]];
    const FLEX_CONTAINER = CANVAS.closest(".llmvis-flex-container");
    const FLEX_CHILD_COUNT = FLEX_CONTAINER.querySelectorAll(":scope > .llmvis-flex-child").length;
    const MAX_SCALE = 1.0;

    var currentScale = 1.0;
    var offsetScale = 1.0;
    var chunks;
    var chunkWidth;
    var xOffset;
    var stoppedScrolling = false;
    var minProb;
    var maxProb;
    var showMoreBoundaries;
    var showPreviousBoundaries;
    var startIndex = 0;
    var fallbackTokenStackTop = 0;
    var oldFallbackTokenStackTops = [0];
    var newFallbackTokenStackTop;

    // Redraw the visualization and refit it to the screen.
    const UPDATE = function(readjust_window = false) {
        // Clear canvas
        // Use offsetScale to make sure entire canvas is cleared
        // including bits outside CTX's scaled dimensions.
        CTX.clearRect(0, 0, CANVAS.width/offsetScale, CANVAS.height/offsetScale);

        const DRAW_RESULT = updateAlternativeTokens(CTX, candidateTokenGroups,
            selectedIndices, fallbackTokens, currentScale, PALETTE, startIndex, fallbackTokenStackTop);
        const FURTHEST_EXTENT = DRAW_RESULT.furthestExtent*offsetScale;

        chunks = DRAW_RESULT.chunks;
        chunkWidth = DRAW_RESULT.chunkWidth;
        xOffset = DRAW_RESULT.xOffset;
        minProb = DRAW_RESULT.minProb;
        maxProb = DRAW_RESULT.maxProb;
        showMoreBoundaries = DRAW_RESULT.showMoreBoundaries;
        showPreviousBoundaries = DRAW_RESULT.showPreviousBoundaries;
        newFallbackTokenStackTop = DRAW_RESULT.fallbackTokenStackTop;

        if (readjust_window) {
            const FLEX_CHILD_WIDTH = window.innerWidth / FLEX_CHILD_COUNT;
            var newWidth = (FURTHEST_EXTENT > window.innerWidth) ? FURTHEST_EXTENT : window.innerWidth;

            if (CANVAS.width != newWidth) {
                CANVAS.width = newWidth;
                updateAlternativeTokens(CTX, candidateTokenGroups,
                    selectedIndices, fallbackTokens, offsetScale, PALETTE, startIndex, fallbackTokenStackTop);
                CANVAS.parentElement.style.width = FLEX_CHILD_WIDTH + "px";
            }
        }
    };

    /*
    alternativeTokensVisualizationAdded is a custom
    property that we assign to the window to prevent
    a new "resize" event being added each time this
    function is run again (such as redrawing for new
    data or comparisons).

    Note that setting the onresize event for the window
    is a bad idea in this case since the environment
    where these visualizations are being displayed may
    already have resize events assigned to the window
    that we do not want to overwrite.
    */
    if (!llmvisVisualizationResizeIds.has(canvasId)) {
        llmvisVisualizationResizeIds.add(canvasId);
        window.addEventListener("resize", function() { UPDATE(true); });
    }

    CANVAS.parentElement.onscroll = function() {
        if (stoppedScrolling) {
            stoppedScrolling = false;
            // Update to remove the tooltip
            UPDATE();
        }
    };

    var hoveringShowMore = false;
    var hoveringShowPrevious = false;

    CANVAS.onmousemove = function(event) {
        if (!chunks) {
            return;
        }

        UPDATE();

        stoppedScrolling = true;

        const mouseX = event.offsetX/offsetScale;
        const mouseY = event.offsetY/offsetScale;
        const CHUNK_LOCATION = Math.floor((mouseX - xOffset) / chunkWidth);
        const CHUNK = chunks.get(CHUNK_LOCATION);

        if (CHUNK) {
            for (TOKEN of CHUNK) {
                if (mouseX >= TOKEN.xStart && mouseX <= TOKEN.xEnd && mouseY >= TOKEN.yStart && mouseY <= TOKEN.yEnd) {
                    CTX.scale(1/offsetScale, 1/offsetScale);
                    drawTooltip([[{text: `Log Probability: ${TOKEN.prob}`, color: "black"}]],
                        event.offsetX, event.offsetY,
                        TOOLTIP_WIDTH, TOOLTIP_HEIGHT,
                        12, CTX);
                    CTX.scale(offsetScale, offsetScale);
                }
            }
        }

        if (showPreviousBoundaries) {
            hoveringShowPrevious = mouseX >= showPreviousBoundaries.x &&
                mouseX <= showPreviousBoundaries.x + showPreviousBoundaries.width &&
                mouseY >= showPreviousBoundaries.y &&
                mouseY <= showPreviousBoundaries.y + showPreviousBoundaries.height;
        }

        if (showMoreBoundaries) {
            hoveringShowMore = mouseX >= showMoreBoundaries.x &&
                mouseX <= showMoreBoundaries.x + showMoreBoundaries.width &&
                mouseY >= showMoreBoundaries.y &&
                mouseY <= showMoreBoundaries.y + showMoreBoundaries.height;
        }
    }

    CANVAS.onmousedown = function() {
        if (hoveringShowMore) {
            startIndex += 128;
            UPDATE(true);
            oldFallbackTokenStackTops.push(newFallbackTokenStackTop);
            fallbackTokenStackTop = newFallbackTokenStackTop;
            CANVAS.parentElement.scrollTo({left: 0});
            hoveringShowMore = false;
        }

        if (hoveringShowPrevious) {
            startIndex -= 128;
            oldFallbackTokenStackTops.pop();
            fallbackTokenStackTop = oldFallbackTokenStackTops[oldFallbackTokenStackTops.length - 1];
            UPDATE(true);
            CANVAS.parentElement.scrollTo({left: CANVAS.width});
            hoveringShowPrevious = false;
        }
    }

    var timer;

    CANVAS.onwheel = function (event) {
        if (timer) {
            clearTimeout(timer);
        }

        // Readjust window when the user has stopped scrolling for
        // performance reasons.
        // Figure this out by having a timer do the readjust which is
        // reset each time an onwheel event is receieved.
        timer = setTimeout(() => UPDATE(true, /* readjust window */), 150);
        currentScale = 1.0 - event.deltaY*SCROLL_SCALE, MAX_SCALE;
        offsetScale *= currentScale;

        if (offsetScale > MAX_SCALE) {
            currentScale = 1.0;
            offsetScale = MAX_SCALE;
        } else {
            UPDATE();
            currentScale = 1.0;
        }
    };

    // Set canvas width to 0 to force a readjust (in the case that
    // this entire function is called multiple times)
    CANVAS.width = 0;
    UPDATE(true /* readjust window */);

    // Draw legend
    LEGEND_CTX.clearRect(0, 0, LEGEND_CANVAS.width, LEGEND_CANVAS.height);

    const LEGEND_PADDING = 40;
    const GRADIENT = LEGEND_CTX.createLinearGradient(LEGEND_PADDING, 0,
        LEGEND_CANVAS.width - LEGEND_PADDING, 0);
    var colorStopIndex = 0;

    GRADIENT.addColorStop(0.0, calculateRgb(minProb, maxProb, minProb, PALETTE));
    GRADIENT.addColorStop(0.5, calculateRgb((minProb + maxProb)/2, maxProb, minProb, PALETTE));
    GRADIENT.addColorStop(1.0, calculateRgb(maxProb, maxProb, minProb, PALETTE));

    LEGEND_CTX.fillStyle = GRADIENT;
    LEGEND_CTX.fillRect(LEGEND_PADDING, LEGEND_PADDING,
        LEGEND_CANVAS.width - LEGEND_PADDING*2,
        LEGEND_CANVAS.height - LEGEND_PADDING*2);

    const LEGEND_MIN_TEXT = `${minProb} (Lowest probability)`;
    const LEGEND_MAX_TEXT = `${maxProb} (Highest Probability)`;
    const LEGEND_MIN_TEXT_MEASUREMENTS = LEGEND_CTX.measureText(LEGEND_MIN_TEXT);
    const LEGEND_MAX_TEXT_MEASUREMENTS = LEGEND_CTX.measureText(LEGEND_MAX_TEXT);
    const LEGEND_MIN_TEXT_HEIGHT = LEGEND_MIN_TEXT_MEASUREMENTS.actualBoundingBoxAscent +
        LEGEND_MIN_TEXT_MEASUREMENTS.actualBoundingBoxDescent;
    const LEGEND_MAX_TEXT_HEIGHT = LEGEND_MAX_TEXT_MEASUREMENTS.actualBoundingBoxAscent +
        LEGEND_MAX_TEXT_MEASUREMENTS.actualBoundingBoxDescent;
    const LEGEND_TEXT_VERTICAL_SPACING = 5;

    LEGEND_CTX.font = "12px DidactGothic";
    LEGEND_CTX.fillStyle = 'rgb(222, 222, 222)';
    LEGEND_CTX.fillText(LEGEND_MIN_TEXT,
        LEGEND_PADDING,
        LEGEND_CANVAS.height - LEGEND_PADDING + LEGEND_TEXT_VERTICAL_SPACING + LEGEND_MIN_TEXT_HEIGHT);
    LEGEND_CTX.fillText(LEGEND_MAX_TEXT,
        LEGEND_CANVAS.width - LEGEND_PADDING - LEGEND_CTX.measureText(LEGEND_MAX_TEXT).width,
        LEGEND_CANVAS.height - LEGEND_PADDING + LEGEND_TEXT_VERTICAL_SPACING + LEGEND_MAX_TEXT_HEIGHT);
}

/**
 * Update the visualization. Called the first time that this visualization
 * is drawn as well as when the screen is resized.
 * @param {Object} ctx The 2D context that should be used for drawing
 *      the visualization.
 * @param {Object} candidateTokenGroups See {@link drawAlternativeTokens}
 * @param {Object} selectedIndices See {@link drawAlternativeTokens}
 * @param {Object} fallbackTokens See {@link drawAlternativeTokens}
 * @param {number} zoom The amount that has been zoomed since the last
 *      update that the visualization needs to be scaled accordingly for.
 * @returns An object with information about the visualization that was just
 * drawn.
 */
function updateAlternativeTokens(ctx, candidateTokenGroups, selectedIndices, fallbackTokens, zoom, palette, startIndex, fallbackTokenStackTop) {
    const STROKE_COLOR = 'rgb(222, 222, 222)';
    const UNSELECTED_TOKEN_COLOR = 'rgb(125, 125, 122)';
    const PADDING = 100;
    const STARTING_X_POSITION = PADDING;
    const STARTING_Y_POSITION = PADDING;
    const X_SPACING = 128;
    const Y_SPACING = 20;
    const FONT_SIZE = 35;
    const CONNECTOR_SPACING = 12;
    const MAX_GROUPS = 64;
    const BUTTON_PLACEMENT_MARGIN = 14;

    var yPosition;
    var xPosition = STARTING_X_POSITION;
    var maxTextWidth;
    var maxProb;
    var minProb;

    ctx.scale(zoom, zoom);
    ctx.strokeStyle = STROKE_COLOR;
    ctx.beginPath();

    for (var group of candidateTokenGroups) {
        for (token of group) {
            const TEXT_WIDTH = ctx.measureText(token.text).width;

            if (!maxTextWidth || TEXT_WIDTH > maxTextWidth) {
                maxTextWidth = TEXT_WIDTH;
            }

            if (!maxProb || token.prob > maxProb) {
                maxProb = token.prob;
            }

            if (!minProb || token.prob < minProb) {
                minProb = token.prob;
            }
        }
    }

    for (var fallbackToken of fallbackTokens) {
        const TEXT_WIDTH = ctx.measureText(fallbackToken.text).width;

        if (!maxTextWidth || TEXT_WIDTH > maxTextWidth) {
            maxTextWidth = TEXT_WIDTH;
        }

        if (!maxProb || fallbackToken.prob > maxProb) {
            maxProb = fallbackToken.prob;
        }

        if (!minProb || fallbackToken.prob < minProb) {
            minProb = fallbackToken.prob;
        }
    }

    const CHUNK_WIDTH = Math.max(maxTextWidth, ctx.measureText("...").width) +
        X_SPACING;

    /*
    Will be used to store the (x, y) position for
    the end of the most recently drawn selected token,
    so that a connector can be drawn from this point.
    */
    var lastCurvePoint;
    var lastSelectedRgb;
    var furthestExtent;
    var chunks = new Map();
    var showPreviousBoundaries;
    var showMoreBoundaries;

    if (startIndex > 0) {
        const BOUNDARIES = drawShowButton(ctx, BUTTON_PLACEMENT_MARGIN, `Show previous...`);
        xPosition = BOUNDARIES.width + X_SPACING;
        showPreviousBoundaries = BOUNDARIES;
    }

    ctx.font = `${FONT_SIZE}px DidactGothic`;

    for (var i = startIndex; i < Math.min(startIndex + MAX_GROUPS, candidateTokenGroups.length); i++) {
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
        var selectedRgb;

        for (j = 0; j < GROUP.length; j++) {
            const TOKEN = GROUP[j];
            const TEXT = TOKEN.text;
            const PROB = TOKEN.prob;
            const MEASUREMENTS = ctx.measureText(TEXT);
            const TEXT_WIDTH = MEASUREMENTS.width;
            const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent +
                MEASUREMENTS.actualBoundingBoxDescent;
            const RGB = calculateRgb(PROB, maxProb, minProb, palette, palette);

            ctx.fillStyle = RGB;

            if (j == SELECTED_INDEX - 1) {
                /*
                This is not the first group of tokens drawn,
                so we need to draw a connector.
                */
                connectorStartX = xPosition;
                connectorEndX = xPosition + TEXT_WIDTH;
                connectorY = yPosition - TEXT_HEIGHT/2;
                foundSelected = true;
                selectedRgb = RGB;
            }

            ctx.fillText(TEXT, xPosition, yPosition);

            const CHUNK_LOCATION = Math.floor((xPosition - STARTING_X_POSITION) / CHUNK_WIDTH);

            if (!chunks.get(CHUNK_LOCATION)) {
                chunks.set(CHUNK_LOCATION, []);
            }

            chunks.get(CHUNK_LOCATION).push({
                xStart: xPosition,
                xEnd: xPosition + TEXT_WIDTH,
                yStart: yPosition - TEXT_HEIGHT,
                yEnd: yPosition,
                prob: PROB
            });
        
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
            const FALLBACK_TOKEN = fallbackTokens[fallbackTokenStackTop++];
            const MEASUREMENTS = ctx.measureText(FALLBACK_TOKEN.text);
            const TEXT_WIDTH = MEASUREMENTS.width;
            const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent +
                MEASUREMENTS.actualBoundingBoxDescent;
            const RGB = calculateRgb(FALLBACK_TOKEN.prob, maxProb, minProb, palette);

            yPosition += TEXT_HEIGHT + Y_SPACING;
            ctx.fillStyle = RGB;
            selectedRgb = RGB;
            ctx.fillText(FALLBACK_TOKEN.text, xPosition, yPosition);
            connectorStartX = xPosition;
            connectorEndX = xPosition + TEXT_WIDTH;
            connectorY = yPosition - TEXT_HEIGHT/2;

            const CHUNK_LOCATION = Math.floor((xPosition - STARTING_X_POSITION) / CHUNK_WIDTH);

            if (!chunks.get(CHUNK_LOCATION)) {
                chunks.set(CHUNK_LOCATION, []);
            }

            chunks.get(CHUNK_LOCATION).push({
                xStart: xPosition,
                xEnd: xPosition + TEXT_WIDTH,
                yStart: yPosition - TEXT_HEIGHT,
                yEnd: yPosition,
                prob: FALLBACK_TOKEN.prob});

            if (TEXT_WIDTH > maxWidth) {
                maxWidth = TEXT_WIDTH;
            }
        }

        // Remember to draw a connector
        ctx.lineWidth = 4;

        if (lastCurvePoint) {
            const CURVE_START_X = lastXPosition + lastMaxWidth + X_SPACING/2;
            const CURVE_START_Y = lastCurvePoint.y;
            const CURVE_MIDDLE_X = CURVE_START_X;
            const CURVE_MIDDLE_Y = connectorY;
            const CURVE_END_X = connectorStartX - CONNECTOR_SPACING;
            const CURVE_END_Y = CURVE_MIDDLE_Y;
            const GRADIENT = ctx.createLinearGradient(CURVE_START_X,
                0, CURVE_END_X, 0);
            
            GRADIENT.addColorStop(0, lastSelectedRgb);
            GRADIENT.addColorStop(1, selectedRgb);

            ctx.strokeStyle = lastSelectedRgb;
            ctx.beginPath();
            ctx.moveTo(lastCurvePoint.x + CONNECTOR_SPACING, lastCurvePoint.y);
            ctx.lineTo(lastXPosition + lastMaxWidth, lastCurvePoint.y);
            ctx.strokeStyle = GRADIENT;
            ctx.bezierCurveTo(
                CURVE_START_X, CURVE_START_Y,
                CURVE_MIDDLE_X, CURVE_MIDDLE_Y,
                CURVE_END_X, CURVE_END_Y 
            )
            ctx.stroke();
        }

        lastCurvePoint = {x: connectorEndX, y: connectorY};
        lastMaxWidth = maxWidth;
        lastXPosition = xPosition;
        lastSelectedRgb = selectedRgb;
        furthestExtent = xPosition + maxWidth + PADDING;
        xPosition += maxWidth + X_SPACING;
    }

    if (startIndex + MAX_GROUPS < candidateTokenGroups.length) {
        const BOUNDARIES = drawShowButton(ctx, xPosition - BUTTON_PLACEMENT_MARGIN,
            `Show next...`);

        furthestExtent = xPosition + BOUNDARIES.width;
        showMoreBoundaries = BOUNDARIES;
    }

    ctx.stroke();
    return {furthestExtent: furthestExtent,
        chunks: chunks,
        chunkWidth: CHUNK_WIDTH,
        xOffset: STARTING_X_POSITION,
        minProb: minProb,
        maxProb: maxProb,
        showPreviousBoundaries: showPreviousBoundaries,
        showMoreBoundaries: showMoreBoundaries,
        fallbackTokenStackTop: fallbackTokenStackTop
    };
}

/**
 * Draw a button to a 2D rendering context that can be used for showing
 * more groups.
 * @param {CanvasRenderingContext2D} ctx The context that should be used
 *      for drawing this button.
 * @param {number} x The x position of this button. The y position will be
 *      centred.
 * @param {string} text The text that should be shown on this button.
 * @returns An object containing the `x` and `y` properties with the x and
 * y positions of the button, respectively. Also contains the `width` and
 * `height` properties which contains the width and height of the button
 * respectively.
 */
function drawShowButton(ctx, x, text) {
    ctx.font = "21px DidactGothic";

    const LOAD_MORE_BUTTON_MARGIN = 4;
    const LOAD_MORE_BUTTON_RADIUS = 100;
    const BUTTON_STR = text;
    const BUTTON_STR_MEASUREMENTS = ctx.measureText(BUTTON_STR);
    const BUTTON_STR_WIDTH = BUTTON_STR_MEASUREMENTS.width;
    const BUTTON_STR_HEIGHT = BUTTON_STR_MEASUREMENTS.actualBoundingBoxAscent +
        BUTTON_STR_MEASUREMENTS.actualBoundingBoxDescent;
    const BUTTON_X = x;
    const BUTTON_Y = ctx.canvas.height/2 + BUTTON_STR_HEIGHT/2;

    ctx.strokeStyle = "rgb(111, 113, 140)";
    ctx.fillStyle = "rgb(111, 113, 140)";
    ctx.beginPath();
    ctx.roundRect(BUTTON_X, BUTTON_Y,
        BUTTON_STR_WIDTH + LOAD_MORE_BUTTON_MARGIN*2,
        BUTTON_STR_HEIGHT + LOAD_MORE_BUTTON_MARGIN*2,
        LOAD_MORE_BUTTON_RADIUS);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = 'rgb(222, 222, 222)';
    ctx.fillText(BUTTON_STR,
        BUTTON_X + LOAD_MORE_BUTTON_MARGIN,
        BUTTON_Y + BUTTON_STR_HEIGHT + LOAD_MORE_BUTTON_MARGIN);

    return {
        x: BUTTON_X,
        y: BUTTON_Y,
        width: BUTTON_STR_WIDTH + LOAD_MORE_BUTTON_MARGIN*2,
        height: BUTTON_STR_HEIGHT + LOAD_MORE_BUTTON_MARGIN*2
    }
}