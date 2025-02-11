/**
 * Draw a Tag Cloud visualization.
 * 
 * @param {Object} units A list of unit objects that should
 *      be visualized by this tag cloud.
 */
function drawTagCloud(units) {
    const TAG_CLOUD_CANVAS = document.getElementById('llmvis-tagcloud-canvas');
    const TAG_CLOUD_CTX = TAG_CLOUD_CANVAS.getContext('2d');

    const TAG_CLOUD_UNIT_COUNT = 12;
    const TAG_CLOUD_SPRAL_ADJUSTMENTS = [
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1]
    ];

    // Sort highest -> lowest
    units.sort(function (a, b) {
        return b.weight - a.weight;
    });

    units = units.slice(0, TAG_CLOUD_UNIT_COUNT);
    const MAX = parseFloat(units[0].weight);
    const MIN = parseFloat(units[units.length - 1].weight);
    // "Shift" all values if there are negative values so that
    // they are positive instead. Do this by creating a value
    // (ADDER) that is added to all weights that will shift the lowest
    // weight to 0 if there are negative weights.
    const DIVISOR = Math.abs(MAX) + (MIN < 0) ? Math.abs(MIN) : 0;
    const ADDER = (MIN < 0) ? Math.abs(MIN) : 0;
    const MIN_FONT_SIZE = 20;
    const MAX_FONT_SIZE = 60;
    const SHOWED_UNITS = units;

    var memory = [];

    // Algorithm:
    // For each of the units that need to be shown,
    // try to draw it at a random x position within
    // the canvas in a roughly centred y position.
    // If this intersects another unit, try a variety
    // of other nearby locations in an approximate
    // spiral shape until we find one that is not
    // intersecting with any other unit.
    for (const UNIT of SHOWED_UNITS) {
        // Calculate size of unit based on its weight
        var fontSize = Math.round(MAX_FONT_SIZE * ((parseFloat(UNIT.weight) + ADDER) / DIVISOR));

        if (fontSize < MIN_FONT_SIZE) {
            fontSize = MIN_FONT_SIZE;
        }

        TAG_CLOUD_CTX.fillStyle = 'white';
        TAG_CLOUD_CTX.font = `${fontSize}px DidactGothic`;

        const MEASUREMENTS = TAG_CLOUD_CTX.measureText(UNIT.text);
        const ASCENT = MEASUREMENTS.actualBoundingBoxAscent;
        const DESCENT = MEASUREMENTS.actualBoundingBoxDescent;
        const FONT_WIDTH = MEASUREMENTS.width;
        const FONT_HEIGHT = ASCENT + DESCENT;
        const ORIGINAL_X = Math.random() * (TAG_CLOUD_CANVAS.width - FONT_WIDTH);
        const ORIGINAL_Y = TAG_CLOUD_CANVAS.height / 2;

        var x = ORIGINAL_X;
        var y = ORIGINAL_Y;
        var spiralIndex = 0;
        var multiplier = 1;

        // Search for a position where this unit can be drawn where it will not intersect
        // with anything.
        while (intersectsWithAnything(x, x + FONT_WIDTH, y - FONT_HEIGHT, y, memory)) {
            const ADJUSTMENT = TAG_CLOUD_SPRAL_ADJUSTMENTS[spiralIndex];
            const X_ADJUST = ADJUSTMENT[0];
            const Y_ADJUST = ADJUSTMENT[1];

            x = ORIGINAL_X + X_ADJUST * multiplier;
            y = ORIGINAL_Y + Y_ADJUST * multiplier;

            if (++spiralIndex >= TAG_CLOUD_SPRAL_ADJUSTMENTS.length) {
                spiralIndex = 0;
                multiplier++;
            }
        }

        TAG_CLOUD_CTX.fillText(UNIT.text, x, y);
        memory.push({
            xStart: x,
            xEnd: x + FONT_WIDTH,
            yStart: y - FONT_HEIGHT,
            yEnd: y
        });
    }
}

/**
 * Returns `true` if the rectangle given by the provided
 * arguments intersects with any of the rectangles contained
 * in some given memory. Used to calculate if two tags in
 * the Tag Cloud visualization are intersecting so that one
 * of the tags can be moved to a new position.
 * @param {*} xStart The starting x position of the requested
 * rectangle. If in doubt, use the x position that this tag
 * is drawn at.
 * @param {*} xEnd The ending x position of the requested
 * rectangle. If in doubt, using the x position that this tag
 * is drawn at + the width of the tag.
 * @param {*} yStart The starting y position of the requested
 * rectangle. If in doubt, use the y position that this tag is
 * drawn at.
 * @param {*} yEnd The ending y position of the requested
 * rectangle. If in doubt, use the y position that this tag is
 * drawn at + the height of the tag.
 * @param {*} memory A list containing previous rectangles
 * that have been drawn where each element is an `Object`
 * containing the attributes: `xStart`, `xEnd`, `yStart` and
 * `yEnd` representing the same values described above.
 * @returns 
 */
function intersectsWithAnything(xStart, xEnd, yStart, yEnd, memory) {
    // TODO: make more efficient

    for (const ITEM of memory) {
        if (xStart <= ITEM.xEnd && xEnd >= ITEM.xStart
            && yStart <= ITEM.yEnd && yEnd >= ITEM.yStart) {
            return true;
        }
    }

    return false;
}