var wordValues;

const FIELD = document.getElementById('llmvis-word-text-field');

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

    lineChartValues = RESULT;
    drawLineChart();
});