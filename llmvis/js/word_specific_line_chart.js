/**
 * Connect a given field to a newly created line chart that will
 * display the corresponding line chart for a typed-in value.
 * 
 * @param {string} canvasId The ID of the canvas that the line chart
 *      should be drawn to.
 * @param {string} fieldId The ID of the field that should be linked
 *      to the new line chart.
 * @param {Object} wordValues The object mapping a text value to the
 *      corresponding line chart data.
 */
function connectFieldToLineChart(canvasId, fieldId, wordValues, xLabel, yLabel) {
    const FIELD = document.getElementById(fieldId);

    // When the user pressed the enter key after editing the text
    // box, find what they have typed in the values and redraw
    // the line chart with the requested values.
    FIELD.addEventListener("keyup", (event) => {
        if (event.key != "Enter") {
            return;
        }

        const RESULT = wordValues[FIELD.value.toString()];

        if (RESULT == undefined) {
            // Does not exist
            // TODO: Add error message
            return;
        }

        drawLineChart(canvasId, RESULT, xLabel, yLabel);
    });
}