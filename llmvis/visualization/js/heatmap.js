const HEATMAP_CANVAS = document.getElementById('llmvis-heatmap-canvas')
const CTX = HEATMAP_CANVAS.getContext("2d")

const SPACING = 30;
const X_INIT = SPACING;
const Y_INIT = 70;
const NEUTRAL_GREY_VALUE = 69;
const HEIGHT_ADJUST = 20;
const MARGIN = 10;
const SMALL_FONT = '12px DidactGothic';
const MEDIUM_FONT = '31px DidactGothic';
const LARGE_FONT = '50px DidactGothic';
const FONT_COLOR = 'white';
const BOTTOM_SPACE = 210;
const KEY_GRADIENT_HEIGHT = 20;
const EXPLANATION_BOX_WIDTH = 160;
const EXPLANATION_BOX_HEIGHT = 112;

var unitSizes = new Map();
var chunkSize;
var chunks = new Map();

// To be defined
var units;
var minWeight;
var maxWeight;

HEATMAP_CANVAS.onmousemove = onMouseMove;

/**
 * Called when the mouse is moved within the heatmap. Used
 * for showing hover effects.
 * @param {MouseEvent} event The event details
 * @returns 
 */
function onMouseMove(event) {
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
    drawHeatmap();

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

    drawExplanation(match, mouseX, mouseY);
}

/**
 * Draw the heatmap visualization. Note that this clears the screen.
 */
function drawHeatmap() {
    CTX.clearRect(0, 0, HEATMAP_CANVAS.width, HEATMAP_CANVAS.height);

    drawUnits();
    drawKey(minWeight, maxWeight);
}

/**
 * Performs calculations to scale the canvas appropriately to fit
 * all the text inside (if needed). Should be run before actually
 * drawing the visualization since scaling the canvas as necessary
 * while drawing each unit causes visual errors.
 */
function calculateCanvasSize() {
    var widthSum = X_INIT;
    var heightSum = Y_INIT;

    for (const unit of units) {
        CTX.font = LARGE_FONT;

        // Calculate (without drawing) the size of the unit
        const measurements = CTX.measureText(unit.text);
        const ascent = measurements.fontBoundingBoxAscent;
        const descent = measurements.fontBoundingBoxDescent;
        const font_width = measurements.width;
        const font_height = ascent + descent;
        const adjusted_font_width = MARGIN * 2 + font_width;
        const adjusted_font_height = MARGIN * 2 + font_height;

        // Cache these calculations to prevent redundant
        // re-calculations of the displayed units when
        // they are actually being drawn.
        unitSizes.set(unit.text, [adjusted_font_width, adjusted_font_height]);
        widthSum += adjusted_font_width + SPACING;

        if (widthSum + adjusted_font_width > HEATMAP_CANVAS.width) {
            widthSum = SPACING;
            heightSum += adjusted_font_height + SPACING;
        }

        if (heightSum > HEATMAP_CANVAS.height - BOTTOM_SPACE) {
            // Units are going out of bounds- resize the canvas
            HEATMAP_CANVAS.height = heightSum + BOTTOM_SPACE;
        }
    }
}

/**
 * Draw an explanation box at a given position
 * containing the details of a given unit. If
 * all the details cannot be fit inside this
 * box, they will be truncated appropriately.
 * @param {Object} unit The unit that an explanation
 *      box will be drawn for
 * @param {number} xPos The x position that the
 *      explanation box should be drawn at
 * @param {number} yPos The y position that the
 *      explanation box should be drawn at
 */
function drawExplanation(unit, xPos, yPos) {
    const BOX_RADIUS = 35;

    // Draw background box
    CTX.fillStyle = 'rgb(92, 92, 92)';
    CTX.beginPath();
    CTX.roundRect(xPos, yPos,
        EXPLANATION_BOX_WIDTH, EXPLANATION_BOX_HEIGHT,
        BOX_RADIUS);
    CTX.fill();

    CTX.font = SMALL_FONT;

    var textXPos = BOX_RADIUS / 2;
    var textYPos = BOX_RADIUS;
    var truncateRest = false;

    const SPACING = 5;
    const NEWLINE_SPACING = 24;

    for (detail of unit.details) {
        const detailName = detail[0];
        const detailValue = detail[1];
        const detailNameText = detailName + ':'

        CTX.font = SMALL_FONT;
        CTX.fillStyle = 'rgb(250, 212, 133)';
        CTX.fillText(detailNameText, xPos + textXPos, yPos + textYPos);

        const nameMeasurement = CTX.measureText(detailNameText);

        textXPos += nameMeasurement.width + SPACING;

        for (const word of detailValue.split(' ')) {
            const wordMeasurement = CTX.measureText(word);
            const ascent = wordMeasurement.fontBoundingBoxAscent;
            const descent = wordMeasurement.fontBoundingBoxDescent;
            const fontWidth = wordMeasurement.width;
            const fontHeight = ascent + descent;

            CTX.font = SMALL_FONT;
            CTX.fillStyle = FONT_COLOR;

            if (textXPos + fontWidth + SPACING > EXPLANATION_BOX_WIDTH - (BOX_RADIUS / 2)) {
                // Text has gone out of bounds- wrap it around to the next line
                textXPos = BOX_RADIUS / 2;
                textYPos += fontHeight + SPACING;
            }

            if (textYPos > EXPLANATION_BOX_HEIGHT - BOX_RADIUS) {
                // Remainder will not fit in the box- truncate it
                CTX.fillText('...', xPos + textXPos, yPos + textYPos);
                truncateRest = true;
                break;
            }

            CTX.fillText(word, xPos + textXPos, yPos + textYPos);
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
 */
function drawUnits() {
    var x = X_INIT;
    var y = Y_INIT;

    for (const unit of units) {
        const sizes = unitSizes.get(unit.text);
        const font_width = sizes[0];
        const font_height = sizes[1];

        if (x + font_width + SPACING > HEATMAP_CANVAS.width) {
            x = X_INIT;
            y += font_height + SPACING;
        }

        // Calculate chunk size if it hasn't been calculated
        // already
        if (chunkSize == undefined) {
            chunkSize = font_height + MARGIN * 2;
        }

        CTX.font = LARGE_FONT;
        CTX.fillStyle = calculateRgb(unit.weight, minWeight, maxWeight);

        // Rounded rectangle
        CTX.beginPath();
        CTX.roundRect(x - MARGIN, y + MARGIN, font_width,
            -(font_height - HEIGHT_ADJUST), 100, 20);
        CTX.fill();

        CTX.fillStyle = FONT_COLOR;
        CTX.fillText(unit.text, x, y);

        // Add this to the unit object so that it can be
        // used for hover/click detection later.
        unit.x_start = x - MARGIN;
        unit.x_end = x + font_width + MARGIN

        // Find the chunk index (the line number that this
        // unit is displayed on) so that it can be stored
        // there for hover/click detection later.
        var chunk_location = Math.floor(y / chunkSize);

        if (chunks.get(chunk_location) == undefined) {
            chunks.set(chunk_location, []);
        }

        chunks.get(chunk_location).push(unit);

        x += font_width + SPACING;
    }
}

/**
 * Draw the key for the bottom of the visualization. This
 * involves a bar with a gradient providing a reference
 * to the user about the color of lower values vs the color
 * of higher values as well as labels to show what the
 * maximum and minimum unit values are.
 */
function drawKey() {
    const Y_POS = HEATMAP_CANVAS.height - (BOTTOM_SPACE / 2);
    const GRADIENT = CTX.createLinearGradient(SPACING, Y_POS,
        HEATMAP_CANVAS.width - SPACING, Y_POS);

    // Beginning of the gradient - blue value
    GRADIENT.addColorStop(0, calculateRgb(minWeight, minWeight, maxWeight));
    // Middle of the gradient - grey value
    GRADIENT.addColorStop(0.5, calculateRgb(0.0, minWeight, maxWeight));
    // End of the gradient - red value
    GRADIENT.addColorStop(1, calculateRgb(maxWeight, minWeight, maxWeight));

    CTX.fillStyle = GRADIENT;
    CTX.fillRect(SPACING, Y_POS, HEATMAP_CANVAS.width - SPACING * 2, KEY_GRADIENT_HEIGHT);

    CTX.fillStyle = FONT_COLOR;
    CTX.font = MEDIUM_FONT;

    // Measure the max weight to calculate how far in from the end of the canvas
    // it should be.
    const measurements = CTX.measureText(maxWeight.toString());

    CTX.fillText(minWeight.toString(), SPACING, Y_POS + KEY_GRADIENT_HEIGHT + SPACING);
    CTX.fillText(maxWeight.toString(), HEATMAP_CANVAS.width - SPACING - measurements.width,
        Y_POS + KEY_GRADIENT_HEIGHT + SPACING);
}

/**
 * Calculate the RGB value that should be used for coloring a
 * unit based on a provided weight.
 * @param {*} weight The weight that should be used for
 *      calculating the RGB value.
 * @returns The CSS-style `rgb(red, green, blue)` RGB value
 *      that the unit should be based on the provided weight.
 */
function calculateRgb(weight) {
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