/**
 * Draw the heatmap visualization. Note that this clears the screen.
 * 
 * @param {string} canvasId The ID of the canvas that this heatmap
 *      should be drawn to.
 * @param {Object} units A list of Unit objects, containing `name`,
 *      `weight` and `details`.
 * @param {number} minWeight The minimum weight out of all units.
 * @param {number} maxWeight The maximum weight out of all units.
 */
function drawHeatmap(canvasId, units, minWeight, maxWeight) {
    const HEATMAP_CANVAS = document.getElementById(canvasId)
    const CTX = HEATMAP_CANVAS.getContext("2d")
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

    var chunkData;
    var chunkSize;
    var chunks;

    HEATMAP_CANVAS.onmousemove = function(event) {
        const mouseX = event.clientX - HEATMAP_CANVAS.offsetLeft;
        const mouseY = event.clientY - HEATMAP_CANVAS.offsetTop;

        // To reduce redundant searching of the unit locations,
        // a chunking-based approach is used. Find the "chunk"
        // that the cursor is in based on its y location. A
        // "chunk" is essentially a line of units. From this,
        // search the narrowed down units to find the one that
        // the cursor is currenty hovering over.
        const chunkLocation = Math.floor(mouseY / chunkSize);
        const chunk = chunks.get(chunkLocation)

        // Mouse movements means that hover box should move accordingly
        // and appear/reappear if the cursor is now hovering/no longer
        // hovering over a unit.
        drawHeatmap(canvasId, units, minWeight, maxWeight);

        // Not hovering over anything
        if (chunk == undefined) {
            return;
        }

        var match;

        // Search the units in this chunk for the one that is being
        // hovered over.
        for (const candidateUnit of chunk) {
            if (mouseX >= candidateUnit.x_start && mouseX <= candidateUnit.x_end) {
                match = candidateUnit;
                break;
            }
        }

        // None of the units in this chunk are being hovered over (might be
        // between units)
        if (match == undefined) {
            return;
        }

        drawExplanation(match, mouseX, mouseY,
            EXPLANATION_BOX_WIDTH, EXPLANATION_BOX_HEIGHT,
            SMALL_FONT, FONT_COLOR, CTX);
    };

    const UNIT_SIZES = calculateCanvasSize(CTX, HEATMAP_CANVAS, units, X_INIT, Y_INIT,
        LARGE_FONT, MARGIN, SPACING, BOTTOM_SPACE);

    CTX.clearRect(0, 0, HEATMAP_CANVAS.width, HEATMAP_CANVAS.height);

    chunkData = drawUnits(CTX, X_INIT, Y_INIT, maxWeight, minWeight, units, UNIT_SIZES, SPACING,
        HEATMAP_CANVAS, MARGIN, LARGE_FONT, FONT_COLOR);
    chunks = chunkData[0];
    chunkSize = chunkData[1];

    drawKey(HEATMAP_CANVAS, CTX, BOTTOM_SPACE, SPACING, MEDIUM_FONT, FONT_COLOR, maxWeight, minWeight);
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
    var unitSizes = new Map();

    for (const unit of units) {
        ctx.font = font;

        // Calculate (without drawing) the size of the unit
        const measurements = ctx.measureText(unit.text);
        const ascent = measurements.fontBoundingBoxAscent;
        const descent = measurements.fontBoundingBoxDescent;
        const font_width = measurements.width;
        const font_height = ascent + descent;
        const adjusted_font_width = margin*2 + font_width;
        const adjusted_font_height = margin*2 + font_height;

        // Cache these calculations to prevent redundant
        // re-calculations of the displayed units when
        // they are actually being drawn.
        unitSizes.set(unit.text, [adjusted_font_width, adjusted_font_height]);
        widthSum += adjusted_font_width + spacing;

        if (widthSum + adjusted_font_width > canvas.width) {
            widthSum = spacing;
            heightSum += adjusted_font_height + spacing;
        }

        if (heightSum > canvas.height - bottomSpace) {
            // Units are going out of bounds- resize the canvas
            canvas.height = heightSum + bottomSpace;
        }
    }

    return unitSizes;
}

/**
 * Draw an explanation box at a given position
 * containing the details of a given unit. If
 * all the details cannot be fit inside this
 * box, they will be truncated appropriately.
 * @param {Object} unit The unit that an explanation
 *      box will be drawn for.
 * @param {number} xPos The x position that the
 *      explanation box should be drawn at.
 * @param {number} yPos The y position that the
 *      explanation box should be drawn at.
 * @param {number} width The width of the explanation
 *      box.
 * @param {number} height The height of the explanation
 *      box.
 * @param {string} font The font that should be used for
 *      the explanation text.
 * @param {string} fontColor The color that should be used
 *      for displaying the font within the explanation box.
 * @param {Object} ctx The context for the heatmap canvas.
 */
function drawExplanation(unit, xPos, yPos, width, height, font, fontColor, ctx) {
    const BOX_RADIUS = 35;
    const SPACING = 5;
    const NEWLINE_SPACING = 24;

    // Draw background box
    ctx.fillStyle = 'rgb(92, 92, 92)';
    ctx.beginPath();
    ctx.roundRect(xPos, yPos,
        width, height,
        BOX_RADIUS);
    ctx.fill();

    ctx.font = font;

    var textXPos = BOX_RADIUS / 2;
    var textYPos = BOX_RADIUS;
    var truncateRest = false;

    for (detail of unit.details) {
        const detailName = detail[0];
        const detailValue = detail[1];
        const detailNameText = detailName + ':'

        ctx.font = font;
        ctx.fillStyle = 'rgb(250, 212, 133)';
        ctx.fillText(detailNameText, xPos + textXPos, yPos + textYPos);

        const nameMeasurement = ctx.measureText(detailNameText);

        textXPos += nameMeasurement.width + SPACING;

        for (const word of detailValue.split(' ')) {
            const wordMeasurement = ctx.measureText(word);
            const ascent = wordMeasurement.fontBoundingBoxAscent;
            const descent = wordMeasurement.fontBoundingBoxDescent;
            const fontWidth = wordMeasurement.width;
            const fontHeight = ascent + descent;

            ctx.font = font;
            ctx.fillStyle = fontColor;

            if (textXPos + fontWidth + SPACING > width - (BOX_RADIUS / 2)) {
                // Text has gone out of bounds- wrap it around to the next line
                textXPos = BOX_RADIUS / 2;
                textYPos += fontHeight + SPACING;
            }

            if (textYPos > height - BOX_RADIUS) {
                // Remainder will not fit in the box- truncate it
                ctx.fillText('...', xPos + textXPos, yPos + textYPos);
                truncateRest = true;
                break;
            }

            ctx.fillText(word, xPos + textXPos, yPos + textYPos);
            textXPos += fontWidth + SPACING;
        }

        if (truncateRest) {
            break;
        }

        textXPos = BOX_RADIUS / 2;
        textYPos += NEWLINE_SPACING;
    }
}

/**
 * Draw the units provided by the code generated by the
 * Python file that reads this JS file. Each unit is shown
 * as a colored box based on its weight that contains the
 * unit's text.
 * 
 * @param {Object} ctx The context of the heatmap canvas.
 * @param {number} xInit The starting x position.
 * @param {number} yInit The starting y position.
 * @param {number} maxWeight The maximum weight out of all the units.
 * @param {number} minWeight The minimum weight out of all the units.
 * @param {Object} units The unit objects that should be drawn.
 * @param {Map} unitSizes A map containing the size of each unit.
 * @param {number} spacing The spacing between each unit.
 * @param {Object} canvas The heatmap canvas.
 * @param {number} margin The size of the margin within each unit rect.
 * @param {string} font The font that should be used for each unit.
 * @param {string} fontColor The color that should be used for rendering
 *      the fonts.
 * @returns A list where the first element is a map that maps each chunk
 *      ID to the units in that chunk for use by the mouse callback and the
 *      second element is the size of each chunk.
 */
function drawUnits(ctx, xInit, yInit, maxWeight, minWeight, units, unitSizes, spacing, canvas, margin, font, fontColor) {
    var x = xInit;
    var y = yInit;
    var chunkSize;
    var chunks = new Map();
    const HEIGHT_ADJUST = 20;

    for (const unit of units) {
        const sizes = unitSizes.get(unit.text);
        const font_width = sizes[0];
        const font_height = sizes[1];

        if (x + font_width + spacing > canvas.width) {
            x = xInit;
            y += font_height + spacing;
        }

        // Calculate chunk size if it hasn't been calculated
        // already
        if (chunkSize == undefined) {
            chunkSize = font_height + margin*2;
        }

        ctx.font = font;
        ctx.fillStyle = calculateRgb(unit.weight, maxWeight, minWeight);

        // Rounded rectangle
        ctx.beginPath();
        ctx.roundRect(x - margin, y + margin, font_width,
            -(font_height - HEIGHT_ADJUST), 100, 20);
        ctx.fill();

        ctx.fillStyle = fontColor;
        ctx.fillText(unit.text, x, y);

        // Add this to the unit object so that it can be
        // used for hover/click detection later.
        unit.x_start = x - margin;
        unit.x_end = x + font_width + margin;

        // Find the chunk index (the line number that this
        // unit is displayed on) so that it can be stored
        // there for hover/click detection later.
        var chunk_location = Math.floor(y / chunkSize);

        if (chunks.get(chunk_location) == undefined) {
            chunks.set(chunk_location, []);
        }

        chunks.get(chunk_location).push(unit);

        x += font_width + spacing;
    }

    return [chunks, chunkSize];
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
 * @param {number} maxWeight The maximum weight out of all the units.
 * @param {number} minWeight The minimum weight out of all the units.
 */
function drawKey(canvas, ctx, bottomSpace, spacing, font, fontColor, maxWeight, minWeight) {
    const Y_POS = canvas.height - (bottomSpace / 2);
    const GRADIENT = ctx.createLinearGradient(spacing, Y_POS,
        canvas.width - spacing, Y_POS);
    const KEY_GRADIENT_HEIGHT = 20;

    // Beginning of the gradient - blue value
    GRADIENT.addColorStop(0, calculateRgb(minWeight, maxWeight, minWeight));
    // Middle of the gradient - grey value
    GRADIENT.addColorStop(0.5, calculateRgb(0.0, maxWeight, minWeight));
    // End of the gradient - red value
    GRADIENT.addColorStop(1, calculateRgb(maxWeight, maxWeight, minWeight));

    ctx.fillStyle = GRADIENT;
    ctx.fillRect(spacing, Y_POS, canvas.width - spacing*2, KEY_GRADIENT_HEIGHT);

    ctx.fillStyle = fontColor;
    ctx.font = font;

    // Measure the max weight to calculate how far in from the end of the canvas
    // it should be.
    const measurements = ctx.measureText(maxWeight.toString());

    ctx.fillText(minWeight.toString(), spacing, Y_POS + KEY_GRADIENT_HEIGHT + spacing);
    ctx.fillText(maxWeight.toString(), canvas.width - spacing - measurements.width,
        Y_POS + KEY_GRADIENT_HEIGHT + spacing);
}

/**
 * Calculate the RGB value that should be used for coloring a
 * unit based on a provided weight.
 * @param {number} weight The weight that should be used for
 *      calculating the RGB value.
 * @param {number} maxWeight The maximum weight out of all the
 *      weights.
 * @param {number} minWeight The minimum weight out of all the
 *      weights.
 * @returns The CSS-style `rgb(red, green, blue)` RGB value
 *      that the unit should be based on the provided weight.
 */
function calculateRgb(weight, maxWeight, minWeight) {
    const NEUTRAL_GREY_VALUE = 69;
    var rgb = [0.0, 0.0, 0.0]

    /* Values near 0 should be closer to white while values
    near the max or min weights should be closer to red
    or to blue respectively.For RGB values this is done
    by keeping red / blue as the max(1.0) and moving the
    other values away from 1.0 accordingly. */
    if (weight < 0.0) {
        // Move from white to blue
        var other_vals = weight / minWeight
        var rgb_value = NEUTRAL_GREY_VALUE + ((255 - NEUTRAL_GREY_VALUE) * other_vals)
        rgb = [rgb_value - (rgb_value * other_vals), rgb_value - (rgb_value * other_vals), rgb_value]
    } else {
        // Move from white to red
        var other_vals = weight / maxWeight
        var rgb_value = NEUTRAL_GREY_VALUE + ((255 - NEUTRAL_GREY_VALUE) * other_vals)
        rgb = [rgb_value, rgb_value - (rgb_value * other_vals), rgb_value - (rgb_value * other_vals)]
    }

    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}