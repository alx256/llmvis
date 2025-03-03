/**
 * Connect all buttons with a given class name to a canvas that
 * will show a unique radar chart for the clicked button on that
 * canvas according to some data.
 * @param {string} canvasId The ID of the canvas that will be used
 *      to draw the radar charts.
 * @param {string} selectorId The name of the class that all
 *      token buttons belong to.
 * @param {Object} tokenValues An object mapping each token to its
 *      corresponding radar chart data.
 */
function connectButtonsToRadarChart(canvasId, selectorId, tokenValues) {
    const SELECTOR = document.getElementById(selectorId);
    // Clear buttons
    SELECTOR.innerHTML = '';

    var selected;
    var hasSelected = false;
    var index = 0;

    for (value of tokenValues) {
        const KEY = value[0];
        const VALUE = value[1];
        // Add a button for each token
        const BUTTON = document.createElement("button");
        BUTTON.classList.add("llmvis-text");
            BUTTON.classList.add("llmvis-token-button");

        const TEXT = document.createTextNode(KEY);
        BUTTON.appendChild(TEXT);
        BUTTON.onclick = function() {
            // Update the buttons to select this button instead,
            // changing the text color.
            selected.classList.remove("selected");
            selected = this;
            selected.classList.add("selected");

            // Show the radar chart for this token
            drawRadarChart(canvasId, VALUE);
        }

        SELECTOR.appendChild(BUTTON);

        if (!hasSelected) {
            selected = BUTTON;
            selected.classList.add("selected");
            selected.onclick();
            hasSelected = true;
        }

        index += 1;
    }
}