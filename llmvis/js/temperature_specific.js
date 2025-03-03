/**
 * Connects temperature controls for temperature specific visualizations
 * to some action that will be carried out when the temperature is
 * changed.
 * @param {Function} func A function that is called each time that the
 *      temperature is adjusted, taking one argument which is the
 *      selected data from `temperatureValues` at the selected temperature
 *      value.
 * @param {string} sliderId The ID of the temperature slider.
 * @param {string} textInputId The ID of the temperature text input.
 * @param {Object} temperatureValues An object mapping each temperature value
 *      to the data that should be visualized when that temperature value is
 *      selected.
 */
function connectTemperatureControlsToVisualization(func, sliderId, textInputId, temperatureValues) {
    const SLIDER = document.getElementById(sliderId);
    const TEXT_INPUT = document.getElementById(textInputId);

    TEXT_INPUT.value = SLIDER.value;

    TEXT_INPUT.oninput = function() {
        TEXT_INPUT.value = TEXT_INPUT.value.replace(/[^\d.-]/g, '');
    }

    TEXT_INPUT.onkeydown = function(event) {
        if (event.key == "Enter") {
            event.preventDefault();
            TEXT_INPUT.value = TEXT_INPUT.value - Math.round(TEXT_INPUT.value%SLIDER.step);
            SLIDER.value = TEXT_INPUT.value;
            SLIDER.oninput();
            return;
        }
    }

    SLIDER.oninput = function() {
        func(temperatureValues[SLIDER.value]);
        TEXT_INPUT.value = SLIDER.value;
    }
}