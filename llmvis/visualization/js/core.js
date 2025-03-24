/**
 * Core JS utilities for all (or just some) visualizations
 */

/**
 * A set containing the IDs of different canvases
 * that are resizeable.
 */
var llmvisVisualizationResizeIds = new Set();

/**
 * Load the necessary fonts that are needed for visualizations.
 */
async function loadFonts() {
    const font_face = new FontFace('DidactGothic',
        'url(https://fonts.gstatic.com/s/didactgothic/v20/ahcfv8qz1zt6hCC5G4F_P4ASlUuYpg.woff2) format(\'woff2\')');
    const font = await font_face.load();
    document.fonts.add(font);
}

/**
 * Draw a tooltip at a given position
 * containing some text data. If
 * all the text cannot be fit inside this
 * box, it will be truncated appropriately.
 * @param {Object} contents The text data that should
 *      be shown inside this tooltip. Should be a list
 *      where each element is another list of objects
 *      for each line of the tooltip. Each object should
 *      have a `text` and `color` field where `text`
 *      represents the raw text data and `color`
 *      represents the color that it should be rendered in.
 * @param {number} xPos The x position that the
 *      explanation box should be drawn at.
 * @param {number} yPos The y position that the
 *      explanation box should be drawn at.
 * @param {number} width The width of the explanation
 *      box.
 * @param {number} height The height of the explanation
 *      box.
 * @param {number} fontSize The size of the font that
 *      should be used for the explanation text.
 * @param {Object} ctx The context for the heatmap canvas.
 */
function drawTooltip(contents, xPos, yPos, width, height, fontSize, ctx) {
    const BOX_RADIUS = 35;
    const HORIZONTAL_SPACING = 5;
    const VERTICAL_SPACING = 4;
    const NEWLINE_SPACING = 6;
    const FREE_SPACE = width - BOX_RADIUS / 2;

    // Flip box if it is on track to going offscreen
    if (xPos + width > ctx.canvas.width) {
        xPos -= width;
    }

    if (yPos + height > ctx.canvas.height) {
        yPos -= height;
    }

    // Draw background box
    ctx.fillStyle = 'rgb(92, 92, 92)';
    ctx.beginPath();
    ctx.roundRect(xPos, yPos,
        width, height,
        BOX_RADIUS);
    ctx.fill();

    ctx.font = `${fontSize}px DidactGothic`;

    var textXPos = BOX_RADIUS / 2;
    var textYPos = BOX_RADIUS;
    var truncateRest = false;

    for (var i = 0; i < contents.length; i++) {
        var line = contents[i];

        for (j = 0; j < line.length; j++) {
            var item = line[j];
            var words = item.text.split(' ');

            for (k = 0; k < words.length; k++) {
                if (truncateRest) {
                    return;
                }

                var word = words[k];

                const wordMeasurement = ctx.measureText(word);
                const ascent = wordMeasurement.fontBoundingBoxAscent;
                const descent = wordMeasurement.fontBoundingBoxDescent;
                const fontWidth = wordMeasurement.width;

                ctx.fillStyle = item.color;

                if (BOX_RADIUS + fontWidth >= FREE_SPACE) {
                    // Text is too long to fit on any single line
                    var remaining = '';

                    while (BOX_RADIUS + ctx.measureText(word).width > FREE_SPACE) {
                        remaining = word.slice(-1) + remaining;
                        word = word.substring(0, word.length - 1);
                    }

                    words.splice(k + 1, 0, remaining);
                    words[k] = word;
                    k -= 1;
                    continue;
                } else if (textXPos + fontWidth + HORIZONTAL_SPACING > FREE_SPACE) {
                    // Text has gone out of bounds- wrap it around to the next line
                    textXPos = BOX_RADIUS / 2;
                    textYPos += fontSize + VERTICAL_SPACING;
                }

                var potentialNextWordY;

                if (k < words.length - 1) {
                    if (textXPos + fontWidth + ctx.measureText(words[k + 1]).width + HORIZONTAL_SPACING > FREE_SPACE) {
                        potentialNextWordY = textYPos + fontSize + VERTICAL_SPACING;
                    }
                } else if (j < line.length - 1 && line.length > 0) {
                    if (textXPos + fontWidth + ctx.measureText(line[j + 1][0]).width + HORIZONTAL_SPACING > FREE_SPACE) {
                        potentialNextWordY = textYPos + fontSize + VERTICAL_SPACING;
                    }
                } else if (i < contents.length - 1 && contents[i + 1].length > 0) {
                    potentialNextWordY = textYPos + fontSize + NEWLINE_SPACING;
                } else {
                    potentialNextWordY = textYPos;
                }

                if (potentialNextWordY > height - BOX_RADIUS) {
                    // Calculate if the next word shown will cause a new line to be started, truncating
                    // it if so.
                    word = word.substring(0, word.length - 3) + "...";
                    truncateRest = true;
                }

                ctx.fillText(word, xPos + textXPos, yPos + textYPos);
                textXPos += fontWidth + HORIZONTAL_SPACING;
            }
        }

        if (truncateRest) {
            break;
        }

        textXPos = BOX_RADIUS / 2;
        textYPos += fontSize + NEWLINE_SPACING;
    }
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
     * @param {Array} palette An array of 3-element arrays representing
     *      RGB values that should be interpolated between. Default is
     *      `pure blue (for lowest values) -> grey (for middle values) ->
     *      pure red (for highest values)`.
     * @returns The CSS-style `rgb(red, green, blue)` RGB value
     *      that the unit should be based on the provided weight.
     */
function calculateRgb(weight, maxWeight, minWeight,
        palette = [[176, 46, 52], [180, 157, 46], [48, 147, 38]]) {
    // Clamp values, warning just for the record. Can occur with
    // a mistake, but precision errors can also cause this.
    if (weight < minWeight) {
        console.warn(`Weight ${weight} is less than minimum weight ${minWeight}`)
        weight = minWeight;
    }

    if (weight > maxWeight) {
        console.warn(`Weight ${weight} is greater than maximum weight ${maxWeight}`)
        weight = maxWeight;
    }

    const NORMALIZED_WEIGHT = (weight - minWeight)/(maxWeight - minWeight);
    const INDEX = (maxWeight != minWeight) ?
        NORMALIZED_WEIGHT*(palette.length-1) :
        (palette.length-1)/2;

    if (Number.isInteger(INDEX)) {
        // INDEX nicely falls on a usable index
        return arrayToRgb(palette[INDEX]);
    }

    if (INDEX > palette.length - 1) {
        // INDEX is above our maximum value.
        // Can occur with slight precision inaccuracies, but just to be safe
        // we alert this.
        console.warn(`Index ${INDEX} fell above maximum value ${palette.length - 1} when calculating RGB values.`);
        return arrayToRgb(palette[palette.length - 1]);
    }

    var rgb = [0, 0, 0];

    const INDEX_LOWER = Math.floor(INDEX);
    const INDEX_UPPER = INDEX_LOWER + 1;
    const DIFF = INDEX - INDEX_LOWER;
    const RGB_LOWER = palette[INDEX_LOWER];
    const RGB_UPPER = palette[INDEX_UPPER];

    // Interpolate between colors
    rgb[0] = (RGB_UPPER[0] - RGB_LOWER[0])*DIFF + RGB_LOWER[0];
    rgb[1] = (RGB_UPPER[1] - RGB_LOWER[1])*DIFF + RGB_LOWER[1];
    rgb[2] = (RGB_UPPER[2] - RGB_LOWER[2])*DIFF + RGB_LOWER[2];

    return arrayToRgb(rgb);
}

function arrayToRgb(arr) {
    return `rgb(${arr[0]}, ${arr[1]}, ${arr[2]})`;
}

/**
 * Calculate the step between consecutive ticks on an axis
 * based on its maximum value and the maximum number of ticks
 * that should occur. Done in a "nice" way that makes it either
 * 1, 2, 5 or 10 multiplied by an appropriate order of magnitude.
 * @param {number} maxVal
 * @param {number} tickCount
 * @returns A "nice" step that can be used to separate consecutive
 * ticks on the axis.
 */
function niceStep(maxVal, tickCount) {
    const UNSCALED_STEP = maxVal/tickCount;
    const OOM = Math.floor(Math.log10(UNSCALED_STEP));
    const POWED = Math.pow(10, OOM);
    const FRAC = parseFloat((UNSCALED_STEP / POWED).toPrecision(12));
    var scaledFrac;

    if (FRAC < 1) {
        scaledFrac = 1;
    } else if (FRAC < 2) {
        scaledFrac = 2
    } else if (FRAC < 5) {
        scaledFrac = 5;
    } else {
        scaledFrac = 10;
    }

    return scaledFrac*POWED;
}

/**
 * Enable a given canvas to be resized when the window is resized.
 * Will try to fill the screen and adjust when there is a resizing.
 * Accounts for comparisons too.
 * @param {Object} canvas The canvas that should be resizable.
 * @param {Function} redrawFunc The function that should redraw the
 *      contents of the canvas when the resizing occurs.
 */
function enableResizing(canvas, redrawFunc) {
    if (!llmvisVisualizationResizeIds.has(canvas.id)) {
        const FLEX_CONTAINER = canvas.closest(".llmvis-flex-container");
        const FLEX_CHILD_COUNT = FLEX_CONTAINER.querySelectorAll(":scope > .llmvis-flex-child").length;
        const RESIZE_FUNC = function() {
            const FLEX_CHILD_WIDTH = window.innerWidth / FLEX_CHILD_COUNT;
            canvas.width = FLEX_CHILD_WIDTH;
            redrawFunc();
        }

        window.addEventListener("resize", RESIZE_FUNC);
        llmvisVisualizationResizeIds.add(canvas.id);
        RESIZE_FUNC();
    }
}