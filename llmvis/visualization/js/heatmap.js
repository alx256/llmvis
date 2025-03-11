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
    const PALETTE = [
        [69, 69, 69],
        [176, 46, 52]
    ];

    var chunkData;
    var chunkSize;
    var chunks;

    const UNIT_SIZES = calculateCanvasSize(CTX, HEATMAP_CANVAS, units, X_INIT, Y_INIT,
        LARGE_FONT, MARGIN, SPACING, BOTTOM_SPACE);

    CTX.clearRect(0, 0, HEATMAP_CANVAS.width, HEATMAP_CANVAS.height);

    chunkData = drawUnits(
        CTX,
        X_INIT,
        Y_INIT,
        maxWeight,
        minWeight,
        units,
        UNIT_SIZES,
        SPACING,
        HEATMAP_CANVAS,
        MARGIN,
        LARGE_FONT,
        FONT_COLOR,
        PALETTE
    );
    chunks = chunkData[0];
    chunkSize = chunkData[1];

    drawKey(
        HEATMAP_CANVAS,
        CTX,
        BOTTOM_SPACE,
        SPACING, MEDIUM_FONT,
        FONT_COLOR,
        PALETTE,
        maxWeight,
        minWeight
    );

    HEATMAP_CANVAS.onmousemove = function (event) {
        const mouseX = event.clientX - RECT.left;
        const mouseY = event.clientY - RECT.top;

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

        if (match.details.length > 0) {
            drawTooltip(match.details.map((d) => [{ text: d[0] + ": ", color: "rgb(30, 30, 30)" }, { text: d[1], color: FONT_COLOR }]),
                mouseX, mouseY,
                EXPLANATION_BOX_WIDTH, EXPLANATION_BOX_HEIGHT,
                12, CTX);
        }
    };

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
            const adjusted_font_width = margin * 2 + font_width;
            const adjusted_font_height = margin * 2 + font_height;

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
    function drawUnits(ctx, xInit, yInit, maxWeight, minWeight, units, unitSizes, spacing, canvas, margin, font, fontColor, palette) {
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
                chunkSize = font_height + margin * 2;
            }

            ctx.font = font;
            ctx.fillStyle = calculateRgb(unit.weight, maxWeight, minWeight, palette);

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
    function drawKey(canvas, ctx, bottomSpace, spacing, font, fontColor, palette, maxWeight, minWeight) {
        const Y_POS = canvas.height - (bottomSpace / 2);
        const GRADIENT = ctx.createLinearGradient(spacing, Y_POS,
            canvas.width - spacing, Y_POS);
        const KEY_GRADIENT_HEIGHT = 20;

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
        const measurements = ctx.measureText(maxWeight.toString());

        ctx.fillText(minWeight.toString(), spacing, Y_POS + KEY_GRADIENT_HEIGHT + spacing);
        ctx.fillText(maxWeight.toString(), canvas.width - spacing - measurements.width,
            Y_POS + KEY_GRADIENT_HEIGHT + spacing);
    }
}