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
                const fontWidth = wordMeasurement.width;

                ctx.fillStyle = item.color;

                if (BOX_RADIUS + fontWidth >= FREE_SPACE) {
                    // Text is too long to fit on any single line
                    var remaining = '';

                    while (BOX_RADIUS + ctx.measureText(word).width > FREE_SPACE) {
                        remaining = word.slice(-1) + remaining;
                        word = word.substring(0, word.length - 1);
                    }

                    if (remaining != "") {
                        words.splice(k + 1, 0, remaining);
                        words[k] = word;
                        k -= 1;
                        continue;
                    }
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
 * Returns `true` if this is running in a Jupyter notebook
 * environment and `false` if not.
 * @returns A boolean that is `true` if this is running in a
 *      Jupyter notebook environment and `false` if not.
 */
function inJupyterNotebook() {
    return typeof isInJupyterNotebook !== "undefined";
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
    if (inJupyterNotebook()) {
        return;
    }

    if (!llmvisVisualizationResizeIds.has(canvas.id)) {
        const FLEX_CONTAINER = canvas.closest(".llmvis-flex-container");
        const FLEX_CHILD_COUNT = FLEX_CONTAINER.querySelectorAll(":scope > .llmvis-flex-child").length;
        const TABS_CONTAINER = document.getElementById("llmvis-tabs-container");
        const VIS_CONTENT = canvas.closest(".llmvis-visualization-content");
        const TOTAL_FLEX = parseInt(TABS_CONTAINER.style.flexGrow) +
            parseInt(VIS_CONTENT.style.flexGrow);
        const OLD_DISPLAY = VIS_CONTENT.style.display;

        if (VIS_CONTENT.style.display == "none") {
            VIS_CONTENT.style.display = "block";
        }

        const RESIZE_FUNC = function() {
            const FLEX_CHILD_WIDTH = window.innerWidth / FLEX_CHILD_COUNT;
            const FLEX_HEIGHT = window.innerHeight*(parseInt(VIS_CONTENT.style.flexGrow) / TOTAL_FLEX);
            const OTHER_CHILDREN = [...canvas.parentNode.children].filter(child => child !== canvas);

            var subtractors = 0;

            // Calculate other HTML elements (such as interactive elements)
            // that we also need to account for.
            for (const CHILD of OTHER_CHILDREN) {
                subtractors += CHILD.offsetHeight;
            }

            canvas.width = FLEX_CHILD_WIDTH;
            canvas.height = FLEX_HEIGHT - subtractors;
            redrawFunc();
        }

        window.addEventListener("resize", RESIZE_FUNC);
        llmvisVisualizationResizeIds.add(canvas.id);
        RESIZE_FUNC();

        VIS_CONTENT.style.display = OLD_DISPLAY;
    }
}

/**
 * A different data type that can be visualized. Available
 * options:
 * 
 * - **CONTINUOUS**: Real-valued numerical data
 * - **CATEGORICAL**: Data belonging to a discrete number of
 * text or numerical categories.
 */
var DataType = {
    CONTINUOUS: 0,
    CATEGORICAL: 1
};

/**
 * Build a new continuous data set.
 * @param {number} start The numerical data point that the data
 *      should start at.
 * @param {number} end The numerical data point that the data
 *      should end with.
 * @param {number} count The number of data points that exist.
 * @returns A `DataType` representing this data.
 */
function continuousData(start, end, count) {
    return {
        type: DataType.CONTINUOUS,
        start: start,
        end: end,
        count: count
    };
}

/**
 * Build a new categorical data set.
 * @param {Array} values An array containing the categorical data.
 * Can be text or numerical data.
 * @returns A `DataType` representing this data.
 */
function categoricalData(values) {
    return {
        type: DataType.CATEGORICAL,
        values: values
    };
}

/**
 * The different positions that an axis can be drawn in.
 * Note: do not think of this as the "direction" of the
 * axis. i.e. `LEFT` means the axis is vertical and on the
 * left side of the screen, not that it is horizontal and
 * points are going in a leftwards direction. Available
 * options:
 * 
 * - **LEFT**: Axis drawn on the left side of the screen.
 * - **RIGHT**: Axis drawn on the right side of the screen.
 * - **TOP**: Axis drawn on the top side of the screen.
 * - **BOTTOM**: Axis drawn on the bottom side of the screen.
 */
var AxisPosition = {
    LEFT: 0,
    RIGHT: 1,
    TOP: 2,
    BOTTOM: 3
};

/**
 * The position of each tick within its local region.
 * Available options:
 * 
 * - **AUTO**: Automatically detect where the tick position
 * should be based on what data is being represented.
 * Specifically, selects `BEGINNING` for continuous data and
 * `CENTER` for categorical data.
 * - **BEGINNING**: The tick should be as left-most as
 * possible.
 * - **CENTER**: The tick should be centered.
 * - **FULL**: Ticks should try to take up as much of the axis
 * as possible. The first tick will be drawn at the very
 * beginning of the axis, while the last will be drawn at the
 * very end. 
 */
var LocalTickPosition = {
    AUTO: 0,
    BEGINNING: 1,
    CENTER: 2,
    FULL: 3
};

/**
 * Draw an axis.
 * @param {CanvasRenderingContext2D} ctx The context that this axis
 *      should be drawn to.
 * @param {number} marginX The x margin for this axis.
 * @param {number} marginY The y margin for this axis.
 * @param {string} color The CSS-style color that should be used for
 *      drawing this axis.
 * @param {Object} data The actual data to be drawn. Must be an object
 *      created with the `continuousData` or `categoricalData` functions,
 *      or must match the same object format that either of this functions
 *      return.
 * @param {AxisPosition} position The position that this axis should be
 *      drawn to.
 * @param {string} label The label that should be shown to describe this
 *      axis. Default is an empty label.
 * @param {LocalTickPosition} tickPosition The local position of each tick.
 * @returns An `Object` with the `min` and `max` properties representing the
 *      calculated minimum and maximum values for this axis, respectively.
 *      Useful in case that the resulting minimum or maximum values of the
 *      axis do not match the minimum or maximum values of the provided data.
 */
function drawAxis(ctx, marginX, marginY, color, data, position, label = "", tickPosition = LocalTickPosition.AUTO) {
    const MARKING_LENGTH = 5;
    const LABEL_SPACING = 9;
    const AXIS_START_POINT_X = marginX;
    const AXIS_START_POINT_Y = ctx.canvas.height - marginY;
    const AXIS_END_POINT_X = ctx.canvas.width - marginX;
    const AXIS_END_POINT_Y = marginY;
    const X_AXIS_Y = AXIS_START_POINT_Y;
    const Y_AXIS_X = AXIS_START_POINT_X;

    if (tickPosition == LocalTickPosition.AUTO) {
        tickPosition = (data.type == DataType.CATEGORICAL) ?
            LocalTickPosition.CENTER :
            LocalTickPosition.BEGINNING;
    }

    ctx.strokeStyle = color;
    ctx.beginPath();

    var startPointX;
    var startPointY;
    var endPointX;
    var endPointY;
    var xMove;
    var yMove;

    switch (position) {
    case AxisPosition.LEFT:
        startPointX = Y_AXIS_X;
        startPointY = AXIS_START_POINT_Y;
        endPointX = Y_AXIS_X;
        endPointY = AXIS_END_POINT_Y;
        xMove = 0;
        yMove = 1;
        break;
    case AxisPosition.RIGHT:
        startPointX = AXIS_END_POINT_X;
        startPointY = AXIS_START_POINT_Y;
        endPointX = AXIS_END_POINT_X;
        endPointY = AXIS_END_POINT_Y;
        xMove = 0;
        yMove = 1;
        break;
    case AxisPosition.TOP:
        startPointX = AXIS_START_POINT_X;
        startPointY = AXIS_END_POINT_Y;
        endPointX = AXIS_END_POINT_X;
        endPointY = AXIS_END_POINT_Y;
        xMove = 1;
        yMove = 0;
        break;
    case AxisPosition.BOTTOM:
        startPointX = AXIS_START_POINT_X;
        startPointY = X_AXIS_Y;
        endPointX = AXIS_END_POINT_X;
        endPointY = X_AXIS_Y;
        xMove = 1;
        yMove = 0;
        break;
    }

    ctx.moveTo(startPointX, startPointY);
    ctx.lineTo(endPointX, endPointY);
    ctx.stroke();

    var step;
    var max;
    var min;
    var dataPointCount;

    switch (data.type) {
    case DataType.CONTINUOUS:
        min = data.start;
        max = data.end;
        step = niceStep(max - min, data.count);
        // Set the minimum to the closest step equal to
        // or below it.
        min = step*Math.floor(min/step);
        // Adjust maximum to be `stepCount` steps away
        // from the minimum, with an additional step for
        // safety.
        dataPointCount = data.count + 1;
        max = min + step*dataPointCount;
        break;
    case DataType.CATEGORICAL:
        min = 0;
        max = data.values.length - 1;
        step = 1;
        dataPointCount = data.values.length;
        break;
    }

    var tickPos = min;
    var screenX = startPointX;
    var screenY = startPointY;

    if (tickPosition == LocalTickPosition.FULL) {
        dataPointCount--;
    }

    var screenStep = (
        (position == AxisPosition.LEFT || position == AxisPosition.RIGHT) ?
            startPointY - endPointY :
            endPointX - startPointX
        )/dataPointCount;

    if (tickPosition == LocalTickPosition.FULL) {
        dataPointCount++;
    }

    ctx.font = "15px DidactGothic";
    ctx.fillStyle = color;

    if (tickPosition == LocalTickPosition.CENTER) {
        screenX += xMove*(screenStep/2);
        screenY -= yMove*(screenStep/2)
    }

    var maxTickTextWidth = -1;
    var maxTickTextHeight = -1;

    // Round to precision to potential floating point problems
    while ((tickPos = parseFloat(tickPos.toPrecision(12))) <= max) {
        var tickLabel;

        switch (data.type) {
        case DataType.CONTINUOUS:
            tickLabel = tickPos.toString();
            break;
        case DataType.CATEGORICAL:
            tickLabel = data.values[tickPos];
            break;
        }

        const MEASUREMENTS = ctx.measureText(tickLabel);
        const TEXT_WIDTH = MEASUREMENTS.width;
        const TEXT_HEIGHT = MEASUREMENTS.actualBoundingBoxAscent +
            MEASUREMENTS.actualBoundingBoxDescent;

        if (maxTickTextWidth == -1 || TEXT_WIDTH > maxTickTextWidth) {
            maxTickTextWidth = TEXT_WIDTH;
        }

        if (maxTickTextHeight == -1 || TEXT_HEIGHT > maxTickTextHeight) {
            maxTickTextHeight = TEXT_HEIGHT;
        }

        ctx.beginPath();

        if (tickPos <= max) {
            ctx.moveTo(screenX, screenY);
            ctx.lineTo(screenX - yMove*MARKING_LENGTH, screenY + xMove*MARKING_LENGTH);

            if (position == AxisPosition.LEFT || position == AxisPosition.RIGHT) {
                ctx.fillText(tickLabel,
                    screenX - TEXT_WIDTH - MARKING_LENGTH,
                    screenY + TEXT_HEIGHT/2
                );
            } else {
                ctx.fillText(tickLabel,
                    screenX - TEXT_WIDTH/2,
                    screenY + TEXT_HEIGHT + MARKING_LENGTH
                )
            }
        }

        tickPos += step;
        screenX += screenStep*xMove;
        screenY -= screenStep*yMove;
        ctx.stroke();
    }

    // Label
    const LABEL_MEASUREMENTS = ctx.measureText(label);
    const LABEL_HEIGHT = LABEL_MEASUREMENTS.actualBoundingBoxAscent + LABEL_MEASUREMENTS.actualBoundingBoxDescent;
    const LABEL_X = (position == AxisPosition.LEFT || position == AxisPosition.RIGHT) ?
        AXIS_START_POINT_X - MARKING_LENGTH - maxTickTextWidth - LABEL_SPACING :
        AXIS_START_POINT_X + (AXIS_END_POINT_X - AXIS_START_POINT_X)/2 - LABEL_MEASUREMENTS.width/2;
    const LABEL_Y = (position == AxisPosition.LEFT || position == AxisPosition.RIGHT) ?
        (AXIS_START_POINT_Y + AXIS_END_POINT_Y) / 2 + LABEL_MEASUREMENTS.width/2 :
        AXIS_START_POINT_Y + LABEL_HEIGHT + MARKING_LENGTH + maxTickTextHeight + LABEL_SPACING;
    ctx.fillStyle = color;

    if (position == AxisPosition.LEFT || position == AxisPosition.RIGHT) {
        ctx.save();
        ctx.translate((LABEL_X  - LABEL_HEIGHT < 0) ? LABEL_HEIGHT : LABEL_X, LABEL_Y);
        ctx.rotate(-90*Math.PI/180);
        ctx.fillStyle = color;
        ctx.fillText(label, 0, 0);
        ctx.restore();
    } else {
        // Draw label right on edge of canvas if it is going off the canvas
        ctx.fillText(label, LABEL_X,
            (LABEL_Y > ctx.canvas.height) ? ctx.canvas.height : LABEL_Y);
    }

    return {
        min: min,
        max: max
    }
}