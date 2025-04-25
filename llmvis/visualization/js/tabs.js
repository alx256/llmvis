var visualizationContentAreas = document.getElementsByClassName('llmvis-visualization-content');
var tabs = document.getElementsByClassName('llmvis-tab');

// Use first visualization by default
var activeVisualization = 0;
tabs[activeVisualization].classList.add('selected');

// Hide all visualizations except the active one
for (var i = 1; i < visualizationContentAreas.length; i++) {
    visualizationContentAreas[i].style.display = 'none';
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
    visualizationContentAreas[activeVisualization].style.display = 'none';
    visualizationContentAreas[visualizationId].style.display = 'block';

    // Select the tab
    tabs[activeVisualization].classList.remove('selected');
    tabs[visualizationId].classList.add('selected');

    activeVisualization = visualizationId;
}