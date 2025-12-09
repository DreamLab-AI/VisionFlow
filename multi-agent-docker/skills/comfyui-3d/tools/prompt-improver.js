#!/usr/bin/env node
/**
 * Prompt Improvement Tool for Text-to-3D Pipeline
 *
 * Analyzes render validation results and suggests improved prompts
 * for retry attempts when initial generation quality is insufficient.
 */

/**
 * Common issues and their prompt fixes
 */
const ISSUE_FIXES = {
    'incomplete_geometry': {
        description: 'Model has missing or incomplete parts',
        fixes: [
            'complete and fully detailed',
            'all parts visible',
            'whole object shown',
            'no cropped or cut-off elements'
        ]
    },
    'poor_texture': {
        description: 'Textures are blurry or inconsistent',
        fixes: [
            'highly detailed surface texture',
            'sharp crisp details',
            'consistent material throughout',
            'well-defined surface patterns'
        ]
    },
    'bad_silhouette': {
        description: 'Object outline is unclear or merged with background',
        fixes: [
            'strong defined silhouette',
            'clear distinct outline',
            'high contrast against white background',
            'well-separated from surroundings'
        ]
    },
    'wrong_angle': {
        description: 'View angle not suitable for 3D reconstruction',
        fixes: [
            'perfect three-quarter view showing front and side',
            'optimal angle revealing depth and form',
            'clear perspective showing three-dimensionality'
        ]
    },
    'complex_background': {
        description: 'Background elements interfering with segmentation',
        fixes: [
            'on pure white background with no other elements',
            'completely isolated subject',
            'clean uncluttered presentation',
            'single object only'
        ]
    },
    'shadows_present': {
        description: 'Shadows causing segmentation issues',
        fixes: [
            'without any shadows or reflections',
            'shadowless studio lighting',
            'even illumination with no dark areas',
            'flat ambient lighting'
        ]
    },
    'multiple_objects': {
        description: 'Multiple objects confusing the segmentation',
        fixes: [
            'single isolated object only',
            'one subject centered in frame',
            'no additional items or props'
        ]
    },
    'thin_features': {
        description: 'Thin or detailed features lost in reconstruction',
        fixes: [
            'with substantial solid forms',
            'chunky bold proportions',
            'thick visible features',
            'robust structural elements'
        ]
    }
};

/**
 * Analyze render images and determine issues
 * In practice, this would use image analysis or user feedback
 */
function analyzeRenderQuality(renderPaths, userFeedback = null) {
    const issues = [];

    // If user feedback provided, parse it
    if (userFeedback) {
        const feedback = userFeedback.toLowerCase();

        if (feedback.includes('missing') || feedback.includes('incomplete')) {
            issues.push('incomplete_geometry');
        }
        if (feedback.includes('blurry') || feedback.includes('texture')) {
            issues.push('poor_texture');
        }
        if (feedback.includes('outline') || feedback.includes('silhouette') || feedback.includes('edge')) {
            issues.push('bad_silhouette');
        }
        if (feedback.includes('angle') || feedback.includes('view')) {
            issues.push('wrong_angle');
        }
        if (feedback.includes('background') || feedback.includes('clutter')) {
            issues.push('complex_background');
        }
        if (feedback.includes('shadow')) {
            issues.push('shadows_present');
        }
        if (feedback.includes('multiple') || feedback.includes('objects')) {
            issues.push('multiple_objects');
        }
        if (feedback.includes('thin') || feedback.includes('detail')) {
            issues.push('thin_features');
        }
    }

    return issues.length > 0 ? issues : ['general_quality'];
}

/**
 * Generate improved prompt based on identified issues
 */
function improvePrompt(originalPrompt, issues) {
    const improvements = [];

    for (const issue of issues) {
        if (ISSUE_FIXES[issue]) {
            // Pick a random fix from available options
            const fixes = ISSUE_FIXES[issue].fixes;
            const fix = fixes[Math.floor(Math.random() * fixes.length)];
            improvements.push(fix);
        }
    }

    // If no specific issues identified, add general quality boosters
    if (improvements.length === 0) {
        improvements.push(
            'highly detailed and realistic',
            'professional product photography style',
            'optimal viewing angle for 3D reconstruction'
        );
    }

    // Rebuild prompt with improvements
    const parts = originalPrompt.split(',').map(p => p.trim());

    // Insert improvements after the main subject (first part)
    const subject = parts[0];
    const rest = parts.slice(1);

    return [subject, ...improvements, ...rest].join(', ');
}

/**
 * Generate multiple prompt variations for parallel testing
 */
function generatePromptVariations(originalPrompt, count = 3) {
    const variations = [];

    // Variation 1: Emphasize clarity
    variations.push(improvePrompt(originalPrompt, ['bad_silhouette', 'shadows_present']));

    // Variation 2: Emphasize geometry
    variations.push(improvePrompt(originalPrompt, ['incomplete_geometry', 'thin_features']));

    // Variation 3: Emphasize isolation
    variations.push(improvePrompt(originalPrompt, ['complex_background', 'multiple_objects']));

    return variations.slice(0, count);
}

/**
 * Suggest segmentation prompt improvements
 */
function improveSegmentationPrompt(originalPrompt, feedback) {
    const simpleTerms = {
        'car': ['vehicle', 'automobile', 'main car'],
        'building': ['structure', 'main building', 'tower'],
        'person': ['figure', 'human', 'character'],
        'animal': ['creature', 'main animal']
    };

    // If original didn't work, try alternative terms
    for (const [base, alternatives] of Object.entries(simpleTerms)) {
        if (originalPrompt.toLowerCase().includes(base)) {
            return alternatives[Math.floor(Math.random() * alternatives.length)];
        }
    }

    // Default: add "main" prefix
    return `main ${originalPrompt}`;
}

/**
 * Full retry recommendation
 */
function getRetryRecommendation(originalPrompt, segmentationPrompt, renderPaths, userFeedback) {
    const issues = analyzeRenderQuality(renderPaths, userFeedback);

    return {
        originalPrompt,
        issues: issues.map(i => ({
            id: i,
            description: ISSUE_FIXES[i]?.description || 'General quality issue'
        })),
        improvedPrompt: improvePrompt(originalPrompt, issues),
        alternativePrompts: generatePromptVariations(originalPrompt),
        improvedSegmentation: improveSegmentationPrompt(segmentationPrompt, userFeedback),
        recommendations: [
            'Consider increasing inference steps for better detail',
            'Try a different random seed',
            'Simplify the subject description if too complex',
            'Ensure subject is a single, well-defined object'
        ]
    };
}

// CLI interface
if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.length < 1) {
        console.log('Usage: prompt-improver.js "<original_prompt>" [--feedback "<user_feedback>"]');
        console.log('\nExample:');
        console.log('  prompt-improver.js "vintage sports car" --feedback "missing wheels"');
        process.exit(1);
    }

    const originalPrompt = args[0];
    const feedbackIndex = args.indexOf('--feedback');
    const userFeedback = feedbackIndex !== -1 ? args[feedbackIndex + 1] : null;

    const recommendation = getRetryRecommendation(
        originalPrompt,
        'object', // default segmentation
        [],
        userFeedback
    );

    console.log('\n=== Prompt Improvement Recommendation ===\n');
    console.log('Original:', originalPrompt);
    console.log('\nIdentified Issues:');
    recommendation.issues.forEach(i => console.log(`  - ${i.description}`));
    console.log('\nImproved Prompt:');
    console.log(`  ${recommendation.improvedPrompt}`);
    console.log('\nAlternative Variations:');
    recommendation.alternativePrompts.forEach((p, i) => console.log(`  ${i + 1}. ${p.substring(0, 80)}...`));
    console.log('\nRecommendations:');
    recommendation.recommendations.forEach(r => console.log(`  - ${r}`));
}

module.exports = {
    analyzeRenderQuality,
    improvePrompt,
    generatePromptVariations,
    improveSegmentationPrompt,
    getRetryRecommendation
};
