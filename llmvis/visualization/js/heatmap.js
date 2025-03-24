/**
 * Draw the heatmap visualization. Note that this clears the screen.
 * 
 * @param {string} canvasId The ID of the canvas that this heatmap
 *      should be drawn to.
 * @param {Object} units A list of Unit objects, containing `name`,
 *      `weight` and `details`.
 * @param {number} minWeight The minimum weight out of all units.
 * @param {number} maxWeight The maximum weight out of all units.
 * @param {number} colorScheme The integer value of the color scheme that
 *      should be used.
 */
function drawHeatmap(canvasId, units, minWeight, maxWeight, colorScheme) {
    const HEATMAP_CANVAS = document.getElementById(canvasId);
    const CTX = HEATMAP_CANVAS.getContext("2d");
    const RECT = HEATMAP_CANVAS.getBoundingClientRect();
    const SPACING = 30;
    const X_INIT = SPACING;
    const Y_INIT = 70;
    const SMALL_FONT = '12px DidactGothic';
    const MEDIUM_FONT = '31px DidactGothic';
    const LARGE_FONT = '50px DidactGothic';
    const MARGIN = 10;
    const BOTTOM_SPACE = 210;
    const FONT_COLOR = 'white';
    const EXPLANATION_BOX_WIDTH = 160;
    const EXPLANATION_BOX_HEIGHT = 112;
    const COLOR_SCHEMES = [
        [[45, 52, 124], [69, 69, 69], [176, 46, 52]],   // Blue -> Red
        [[176, 46, 52], [69, 69, 69], [45, 52, 124]],   // Red -> Blue
        [[48, 147, 38], [69, 69, 69], [176, 46, 52]],   // Green -> Red
        [[176, 46, 52], [69, 69, 69], [48, 147, 38]]    // Red -> Green
    ];
    const COLD_VALUE = COLOR_SCHEMES[colorScheme][0];
    const NEUTRAL_VALUE = COLOR_SCHEMES[colorScheme][1];
    const HOT_VALUE = COLOR_SCHEMES[colorScheme][2];
    const FULL_PALETTE = [COLD_VALUE, NEUTRAL_VALUE, HOT_VALUE];
    const POSITIVE_PALETTE = [NEUTRAL_VALUE, HOT_VALUE];
    const NEGATIVE_PALETTE = [COLD_VALUE, NEUTRAL_VALUE];

    var palette;

    if (minWeight < 0.0 && maxWeight > 0.0) {
        const MAX_ABS_VALUE = Math.max(Math.abs(minWeight), Math.abs(maxWeight));

        palette = FULL_PALETTE;
        minWeight = -MAX_ABS_VALUE;
        maxWeight = MAX_ABS_VALUE;
    } else if (minWeight < 0.0) {
        palette = NEGATIVE_PALETTE;
    } else {
        palette = POSITIVE_PALETTE;
    }

    const UNIT_SIZES = calculateCanvasSize(CTX, HEATMAP_CANVAS, units, X_INIT, Y_INIT,
        LARGE_FONT, MARGIN, SPACING, BOTTOM_SPACE);
    const RECT_POSITIONS = UNIT_SIZES.rectPositions;

    CTX.clearRect(0, 0, HEATMAP_CANVAS.width, HEATMAP_CANVAS.height);

    drawUnits(
        CTX,
        maxWeight,
        minWeight,
        UNIT_SIZES,
        MARGIN,
        LARGE_FONT,
        FONT_COLOR,
        palette
    );

    drawKey(
        HEATMAP_CANVAS,
        CTX,
        BOTTOM_SPACE,
        SPACING, SMALL_FONT,
        FONT_COLOR,
        palette,
        maxWeight,
        minWeight
    );

    HEATMAP_CANVAS.onmousemove = function (event) {
        const mouseX = event.clientX - RECT.left;
        const mouseY = event.clientY - RECT.top;

        // Mouse movements means that hover box should move accordingly
        // and appear/reappear if the cursor is now hovering/no longer
        // hovering over a unit.
        drawHeatmap(canvasId, units, minWeight, maxWeight, colorScheme);

        var match;

        // TODO: Potentially re-introduce chunk-based approach
        for (const RECT_POSITION of RECT_POSITIONS) {
            if (mouseX >= RECT_POSITION.x && mouseX <= RECT_POSITION.x + RECT_POSITION.width &&
                    mouseY >= RECT_POSITION.y - UNIT_SIZES.height && mouseY <= RECT_POSITION.y) {
                match = RECT_POSITION.unit;
                break;
            }
        }

        // None of the units are being hovered over (might be
        // between units)
        if (match == undefined) {
            return;
        }

        if (match.details.length > 0) {
            drawTooltip(match.details.map((d) => [{ text: d[0] + ": ", color: "rgb(30, 30, 30)" }, { text: d[1], color: FONT_COLOR }]),
                mouseX, mouseY,
                EXPLANATION_BOX_WIDTH, EXPLANATION_BOX_HEIGHT,
                12, CTX);
        }
    };
}

/**
     * Performs calculations to scale the canvas appropriately to fit
     * all the text inside (if needed). Should be run before actually
     * drawing the visualization since scaling the canvas as necessary
     * while drawing each unit causes visual errors.
     * 
     * @param {Object} ctx The context for the heatmap canvas.
     * @param {Object} canvas The heatmap canvas.
     * @param {number} xInit The starting x position.
     * @param {number} yInit The starting y position.
     * @param {string} font The font that should be used for displaying the
     *      units.
     * @param {number} margin The margin that should be used for each unit
     *      rect.
     * @param {number} spacing The spacing between units.
     * @param {number} bottomSpace The space between the bottom of the canvas
     *      and the units. Canvas will be resized if units fall below this.
     * @returns A map that maps each unit to its calculated size so that this
     *      can be cached and used later.
     */
function calculateCanvasSize(ctx, canvas, units, xInit, yInit, font, margin, spacing, bottomSpace) {
    var widthSum = xInit;
    var heightSum = yInit;
    var lastRectStart = xInit;
    var rectPositions = [];
    var wordPositions = [];

    for (const unit of units) {
        const WORDS = unit.text.split(" ");
        var height = undefined;
        var lineWidth = 0;

        ctx.font = font;

        for (const WORD of WORDS) {
            // Calculate (without drawing) the size of the unit
            const measurements = ctx.measureText(WORD);
            const ascent = measurements.fontBoundingBoxAscent;
            const descent = measurements.fontBoundingBoxDescent;
            const font_width = measurements.width;
            const font_height = ascent + descent;
            const adjusted_font_width = margin * 2 + font_width;
            const adjusted_font_height = margin * 2 + font_height;

            if (widthSum + adjusted_font_width > canvas.width && WORDS.length != 1) {
                rectPositions.push({
                    unit: unit,
                    x: lastRectStart,
                    y: heightSum,
                    width: lineWidth - spacing,
                    weight: unit.weight,
                });
                widthSum = spacing;
                lastRectStart = widthSum;
                lineWidth = 0;
                heightSum += adjusted_font_height + spacing;
            }

            wordPositions.push({
                x: widthSum,
                y: heightSum,
                word: WORD
            });

            // Add this to the unit object so that it can be
            // used for hover/click detection later.
            widthSum += adjusted_font_width + spacing;
            lineWidth += adjusted_font_width + spacing;

            if (heightSum > canvas.height - bottomSpace) {
                // Units are going out of bounds- resize the canvas
                canvas.height = heightSum + bottomSpace;
                ctx.font = font;
            }

            if (height == undefined) {
                height = adjusted_font_height;
            }
        }

        rectPositions.push({
            unit: unit,
            x: lastRectStart,
            y: heightSum,
            width: lineWidth - spacing,
            weight: unit.weight,
        });
        lastRectStart = widthSum;
    }

    // Cache these calculations to prevent redundant
    // re-calculations of the displayed units when
    // they are actually being drawn.
    return {
        rectPositions: rectPositions,
        wordPositions: wordPositions,
        height: height
    };
}

/**
 * Draw the units provided by the code generated by the
 * Python file that reads this JS file. Each unit is shown
 * as a colored box based on its weight that contains the
 * unit's text.
 * 
 * @param {Object} ctx The context of the heatmap canvas.
 * @param {number} maxWeight The maximum weight out of all the units.
 * @param {number} minWeight The minimum weight out of all the units.
 * @param {Map} unitSizes A map containing the size of each unit.
 * @param {number} margin The size of the margin within each unit rect.
 * @param {string} font The font that should be used for each unit.
 * @param {string} fontColor The color that should be used for rendering
 *      the fonts.
 * @param {Array} palette The palette that should be used for colouring.
 */
function drawUnits(ctx, maxWeight, minWeight, unitSizes, margin, font, fontColor, palette) {
    const HEIGHT_ADJUST = 20;
    const RECT_POSITIONS = unitSizes.rectPositions;
    const WORD_POSITIONS = unitSizes.wordPositions;
    const HEIGHT = unitSizes.height;

    for (const RECT_POSITION of RECT_POSITIONS) {
        const X = RECT_POSITION.x;
        const Y = RECT_POSITION.y;
        const WIDTH = RECT_POSITION.width;
        const WEIGHT = RECT_POSITION.weight;

        ctx.font = font;
        ctx.fillStyle = calculateRgb(WEIGHT, maxWeight, minWeight, palette);

        // Rounded rectangle
        ctx.beginPath();
        ctx.roundRect(X - margin, Y + margin,
            WIDTH,
            HEIGHT_ADJUST - HEIGHT,
            100);
        ctx.fill();
    }

    for (const WORD_POSITION of WORD_POSITIONS) {
        const X = WORD_POSITION.x;
        const Y = WORD_POSITION.y;
        const WORD = WORD_POSITION.word;

        ctx.fillStyle = fontColor;
        ctx.fillText(WORD, X, Y);
    }
}

/**
 * Draw the key for the bottom of the visualization. This
 * involves a bar with a gradient providing a reference
 * to the user about the color of lower values vs the color
 * of higher values as well as labels to show what the
 * maximum and minimum unit values are.
 * 
 * @param {Object} canvas The heatmap canvas.
 * @param {Object} ctx The heatmap canvas' context.
 * @param {number} bottomSpace The space between the bottom of the
 *      canvas and the last row of units. Any units pass this will
 *      expand the canvas.
 * @param {number} spacing The spacing between each unit.
 * @param {string} font The font that should be used for displaying
 *      the units.
 * @param {string} fontColor The color that should be used when
 *      drawing the font.
 * @param {list} palette The palette that should be used for colouring.
 * @param {number} maxWeight The maximum weight out of all the units.
 * @param {number} minWeight The minimum weight out of all the units.
 */
function drawKey(canvas, ctx, bottomSpace, spacing, font, fontColor, palette, maxWeight, minWeight) {
    const Y_POS = canvas.height - (bottomSpace / 2);
    const GRADIENT = ctx.createLinearGradient(spacing, Y_POS,
        canvas.width - spacing, Y_POS);
    const KEY_GRADIENT_HEIGHT = 20;
    const START_TEXT = `${minWeight} (Lowest value)`
    const END_TEXT = `${maxWeight} (Highest value)`

    // Beginning of the gradient - blue value
    GRADIENT.addColorStop(0, calculateRgb(minWeight, maxWeight, minWeight, palette));
    // Middle of the gradient - grey value
    GRADIENT.addColorStop(0.5, calculateRgb((minWeight+maxWeight)/2, maxWeight, minWeight, palette));
    // End of the gradient - red value
    GRADIENT.addColorStop(1, calculateRgb(maxWeight, maxWeight, minWeight, palette));

    ctx.fillStyle = GRADIENT;
    ctx.fillRect(spacing, Y_POS, canvas.width - spacing * 2, KEY_GRADIENT_HEIGHT);

    ctx.fillStyle = fontColor;
    ctx.font = font;

    // Measure the max weight to calculate how far in from the end of the canvas
    // it should be.
    const measurements = ctx.measureText(START_TEXT);

    ctx.fillText(START_TEXT, spacing, Y_POS + KEY_GRADIENT_HEIGHT + spacing);
    ctx.fillText(END_TEXT, canvas.width - spacing - measurements.width,
        Y_POS + KEY_GRADIENT_HEIGHT + spacing);
}