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