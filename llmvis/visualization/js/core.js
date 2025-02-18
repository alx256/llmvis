/**
 * Core JS utilities for all (or just some) visualizations
 */

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
    const SPACING = 5;
    const NEWLINE_SPACING = 24;
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

    for (i = 0; i < contents.length; i++) {
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
                const fontHeight = ascent + descent;

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
                } else if (textXPos + fontWidth + SPACING > FREE_SPACE) {
                    // Text has gone out of bounds- wrap it around to the next line
                    textXPos = BOX_RADIUS / 2;
                    textYPos += fontHeight + SPACING;
                }

                var potentialNextWordY

                if (k < words.length - 1) {
                    if (textXPos + fontWidth + ctx.measureText(words[k + 1]).width + SPACING > FREE_SPACE) {
                        potentialNextWordY = textYPos + fontHeight + SPACING;
                    }
                } else if (j < line.length - 1 && line.length > 0) {
                    if (textXPos + fontWidth + ctx.measureText(line[j + 1][0]).width + SPACING > FREE_SPACE) {
                        potentialNextWordY = textYPos + fontHeight + SPACING;
                    }
                } else if (i < contents.length - 1 && contents[i + 1].length > 0) {
                    potentialNextWordY = textYPos + fontHeight + SPACING;
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
                textXPos += fontWidth + SPACING;
            }
        }

        if (truncateRest) {
            break;
        }

        textXPos = BOX_RADIUS / 2;
        textYPos += NEWLINE_SPACING;
    }
}