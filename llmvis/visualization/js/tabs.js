const VISUALIZATION_CONTENT_AREAS = document.getElementsByClassName('llmvis-visualization-content');
const TABS = document.getElementsByClassName('llmvis-tab');

// Use first visualization by default
var activeVisualization = 0;
TABS[activeVisualization].classList.add('selected');

// Hide all visualizations except the active one
for (i = 1; i < VISUALIZATION_CONTENT_AREAS.length; i++) {
    VISUALIZATION_CONTENT_AREAS[i].style.display = 'none';
}

/**
 * Open a visualization with a provided `visualizationId`, hiding the
 * currently selected visualization and showing the new one.
 * @param {*} visualizationId The ID of the visualization that should
 * be shown. This corresponds to the tab that opens this visualization's
 * position in the list of tabs, so the first tab will have
 * `visualizationId` = `0`, second tab will have `visualizationId` =
 * `1` and so on.
 */
function openVisualization(visualizationId) {
    VISUALIZATION_CONTENT_AREAS[activeVisualization].style.display = 'none';
    VISUALIZATION_CONTENT_AREAS[visualizationId].style.display = 'block';

    // Select the tab
    TABS[activeVisualization].classList.remove('selected');
    TABS[visualizationId].classList.add('selected');

    activeVisualization = visualizationId;
}