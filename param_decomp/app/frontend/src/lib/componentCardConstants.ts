/**
 * Shared constants for component card displays.
 * Centralizes magic numbers to ensure consistency across ComponentNodeCard and ActivationContextsViewer.
 */

export const COMPONENT_CARD_CONSTANTS = {
    /** Number of correlations per page */
    CORRELATIONS_PAGE_SIZE: 10,

    /** Number of dataset attributions per page */
    DATASET_ATTRIBUTIONS_PAGE_SIZE: 4,

    /** Number of prompt attributions per page */
    PROMPT_ATTRIBUTIONS_PAGE_SIZE: 4,
} as const;
