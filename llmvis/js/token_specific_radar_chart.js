/**
 * Connect all buttons with a given class name to a canvas that
 * will show a unique radar chart for the clicked button on that
 * canvas according to some data.
 * @param {string} canvasId The ID of the canvas that will be used
 *      to draw the radar charts.
 * @param {string} buttonClassName The name of the class that all
 *      token buttons belong to.
 * @param {Object} tokenValues An object mapping each token to its
 *      corresponding radar chart data.
 */
function connectButtonsToRadarChart(canvasId, buttonClassName, tokenValues) {
    const BUTTONS = document.getElementsByClassName(buttonClassName);

    for (button of BUTTONS) {
        button.onclick = function() {
            // Show the radar chart for this token
            drawRadarChart(canvasId, tokenValues[this.innerHTML]);
        }
    }
}