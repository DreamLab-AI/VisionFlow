#!/usr/bin/env node
/**
 * ComfyUI Text-to-3D Pipeline Tool
 *
 * Orchestrates the full pipeline:
 * 1. Expand user prompt to FLUX2-optimized format
 * 2. Submit Phase 1 (FLUX2 image generation)
 * 3. Transfer image for Phase 2
 * 4. Submit Phase 2 (SAM3D reconstruction)
 * 5. Import to Blender for validation renders
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const net = require('net');

// Configuration
const COMFYUI_HOST = process.env.COMFYUI_HOST || '192.168.0.51';
const COMFYUI_PORT = parseInt(process.env.COMFYUI_PORT || '8188');
const BLENDER_HOST = process.env.BLENDER_HOST || 'localhost';
const BLENDER_PORT = parseInt(process.env.BLENDER_PORT || '9876');

const TEMPLATE_DIR = path.join(__dirname, '..', 'templates');
const COMFYUI_OUTPUT = process.env.COMFYUI_OUTPUT || '/home/devuser/workspace/project/multi-agent-docker/comfyui/storage-output';
const COMFYUI_INPUT = process.env.COMFYUI_INPUT || '/home/devuser/workspace/project/multi-agent-docker/comfyui/storage-input';

/**
 * FLUX2 Prompt Expansion
 * Transforms simple descriptions into detailed, optimized prompts
 */
function expandPromptForFlux2(simplePrompt, orientation = 'landscape') {
    // Extract key subject from prompt
    const subject = simplePrompt.trim();

    // Build FLUX2-optimized prompt with required elements
    const expandedParts = [
        subject,
        'with fine details and realistic textures',
        'viewed from three-quarter angle',
        'shot on Hasselblad X2D 100C with 45mm f/3.5 lens',
        'soft diffused studio lighting with even illumination',
        'infinite depth of field with everything in sharp focus',
        'centered composition',
        'isolated on clean white background',
        'without cast shadows'
    ];

    return expandedParts.join(', ');
}

/**
 * Distill prompt for segmentation
 * Extracts simple noun for SAM3/BiRefNet
 */
function distillForSegmentation(prompt) {
    // Common object mappings
    const mappings = {
        'car': ['car', 'vehicle', 'automobile', 'truck', 'sedan', 'sports car'],
        'building': ['building', 'tower', 'skyscraper', 'house', 'castle', 'architecture'],
        'chair': ['chair', 'armchair', 'seat', 'throne', 'stool'],
        'table': ['table', 'desk', 'furniture'],
        'person': ['person', 'human', 'man', 'woman', 'figure'],
        'animal': ['animal', 'dog', 'cat', 'horse', 'creature'],
        'plant': ['plant', 'tree', 'flower', 'vegetation'],
        'weapon': ['sword', 'weapon', 'blade', 'axe', 'gun']
    };

    const lowerPrompt = prompt.toLowerCase();

    for (const [simple, keywords] of Object.entries(mappings)) {
        for (const keyword of keywords) {
            if (lowerPrompt.includes(keyword)) {
                return simple;
            }
        }
    }

    // Default: extract first noun-like word
    const words = prompt.split(/\s+/);
    return words[0].toLowerCase().replace(/[^a-z]/g, '') || 'object';
}

/**
 * Load and customize workflow template
 */
function loadTemplate(templateName, replacements) {
    const templatePath = path.join(TEMPLATE_DIR, templateName);
    let template = fs.readFileSync(templatePath, 'utf8');

    for (const [key, value] of Object.entries(replacements)) {
        const placeholder = `{{${key}}}`;
        template = template.replace(new RegExp(placeholder, 'g'), value);
    }

    return JSON.parse(template);
}

/**
 * Submit workflow to ComfyUI
 */
async function submitToComfyUI(workflow) {
    return new Promise((resolve, reject) => {
        const payload = JSON.stringify({ prompt: workflow });

        const options = {
            hostname: COMFYUI_HOST,
            port: COMFYUI_PORT,
            path: '/prompt',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(payload)
            }
        };

        const req = http.request(options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(data));
                } catch (e) {
                    reject(new Error(`Invalid JSON response: ${data}`));
                }
            });
        });

        req.on('error', reject);
        req.write(payload);
        req.end();
    });
}

/**
 * Check ComfyUI queue/history for completion
 */
async function waitForCompletion(promptId, timeoutMs = 600000) {
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
        try {
            const history = await new Promise((resolve, reject) => {
                http.get(`http://${COMFYUI_HOST}:${COMFYUI_PORT}/history/${promptId}`, (res) => {
                    let data = '';
                    res.on('data', chunk => data += chunk);
                    res.on('end', () => resolve(JSON.parse(data)));
                }).on('error', reject);
            });

            if (history[promptId]) {
                const status = history[promptId].status;
                if (status && status.completed) {
                    return history[promptId];
                }
                if (status && status.status_str === 'error') {
                    throw new Error(`Workflow failed: ${JSON.stringify(status)}`);
                }
            }
        } catch (e) {
            // Queue might not have the item yet
        }

        await new Promise(r => setTimeout(r, 5000)); // Poll every 5 seconds
    }

    throw new Error('Workflow timeout');
}

/**
 * Send command to Blender MCP
 */
async function sendToBlender(command) {
    return new Promise((resolve, reject) => {
        const client = new net.Socket();
        let responseData = '';

        client.setTimeout(60000);

        client.connect(BLENDER_PORT, BLENDER_HOST, () => {
            client.write(JSON.stringify(command));
        });

        client.on('data', (data) => {
            responseData += data.toString();
            try {
                const response = JSON.parse(responseData);
                client.destroy();
                resolve(response);
            } catch (e) {
                // Incomplete JSON, wait for more
            }
        });

        client.on('error', reject);
        client.on('timeout', () => {
            client.destroy();
            reject(new Error('Blender connection timeout'));
        });
    });
}

/**
 * Main pipeline execution
 */
async function runPipeline(userPrompt, options = {}) {
    const {
        orientation = 'landscape',
        filenamePrefix = 'Generated3D',
        skipBlenderValidation = false
    } = options;

    console.log('=== ComfyUI Text-to-3D Pipeline ===\n');

    // Step 1: Expand prompt
    console.log('Step 1: Expanding prompt for FLUX2...');
    const flux2Prompt = expandPromptForFlux2(userPrompt, orientation);
    const segmentationPrompt = distillForSegmentation(userPrompt);
    console.log(`  FLUX2 Prompt: ${flux2Prompt.substring(0, 100)}...`);
    console.log(`  Segmentation Prompt: ${segmentationPrompt}\n`);

    // Step 2: Prepare Phase 1 workflow
    console.log('Step 2: Preparing Phase 1 (FLUX2 Generation)...');
    const seed = Math.floor(Math.random() * 1e15);
    const dimensions = orientation === 'portrait'
        ? { width: 832, height: 1248 }
        : { width: 1248, height: 832 };

    const phase1Workflow = loadTemplate('flux2-phase1-generate.json', {
        FLUX2_PROMPT: flux2Prompt,
        SEED: seed.toString(),
        FILENAME_PREFIX: filenamePrefix
    });

    // Update dimensions
    phase1Workflow['79'].inputs.width = dimensions.width;
    phase1Workflow['79'].inputs.height = dimensions.height;

    // Step 3: Submit Phase 1
    console.log('Step 3: Submitting Phase 1 to ComfyUI...');
    const phase1Result = await submitToComfyUI(phase1Workflow);
    console.log(`  Prompt ID: ${phase1Result.prompt_id}`);

    console.log('  Waiting for Phase 1 completion...');
    const phase1History = await waitForCompletion(phase1Result.prompt_id);
    console.log('  Phase 1 complete!\n');

    // Step 4: Find generated image
    const outputs = phase1History.outputs || {};
    let generatedImage = null;
    for (const nodeOutput of Object.values(outputs)) {
        if (nodeOutput.images && nodeOutput.images.length > 0) {
            generatedImage = nodeOutput.images[0].filename;
            break;
        }
    }

    if (!generatedImage) {
        throw new Error('No image generated in Phase 1');
    }
    console.log(`Step 4: Generated image: ${generatedImage}\n`);

    // Step 5: Copy image to input directory
    console.log('Step 5: Transferring image for Phase 2...');
    const sourcePath = path.join(COMFYUI_OUTPUT, generatedImage);
    const destPath = path.join(COMFYUI_INPUT, generatedImage);

    // Note: May need sudo for root-owned directories
    try {
        fs.copyFileSync(sourcePath, destPath);
        console.log(`  Copied to: ${destPath}\n`);
    } catch (e) {
        console.log(`  Warning: Direct copy failed, may need sudo: ${e.message}\n`);
    }

    // Step 6: Prepare and submit Phase 2
    console.log('Step 6: Preparing Phase 2 (SAM3D Reconstruction)...');
    const phase2Workflow = loadTemplate('flux2-phase2-sam3d-rmbg.json', {
        INPUT_IMAGE: generatedImage
    });

    console.log('Step 7: Submitting Phase 2 to ComfyUI...');
    const phase2Result = await submitToComfyUI(phase2Workflow);
    console.log(`  Prompt ID: ${phase2Result.prompt_id}`);

    console.log('  Waiting for Phase 2 completion (this may take several minutes)...');
    const phase2History = await waitForCompletion(phase2Result.prompt_id, 900000); // 15 min timeout
    console.log('  Phase 2 complete!\n');

    // Step 8: Locate output files
    const outputFiles = {
        mesh: path.join(COMFYUI_OUTPUT, 'mesh.glb'),
        meshTextured: path.join(COMFYUI_OUTPUT, 'mesh_textured.glb'),
        gaussian: path.join(COMFYUI_OUTPUT, 'gaussian.ply'),
        pointcloud: path.join(COMFYUI_OUTPUT, 'pointcloud.ply')
    };

    console.log('Step 8: Generated 3D outputs:');
    for (const [name, filepath] of Object.entries(outputFiles)) {
        try {
            const stats = fs.statSync(filepath);
            console.log(`  ${name}: ${filepath} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`);
        } catch {
            console.log(`  ${name}: Not found`);
        }
    }
    console.log();

    // Step 9: Blender validation (optional)
    if (!skipBlenderValidation) {
        console.log('Step 9: Blender validation renders...');
        try {
            // Import model
            await sendToBlender({
                type: 'import_model',
                params: {
                    file_path: outputFiles.meshTextured || outputFiles.mesh,
                    format: 'gltf'
                }
            });
            console.log('  Model imported to Blender');

            // Execute validation script
            const validationScript = fs.readFileSync(
                path.join(TEMPLATE_DIR, 'blender-validation.py'),
                'utf8'
            );
            await sendToBlender({
                type: 'execute_code',
                params: { code: validationScript }
            });
            console.log('  Validation renders complete');
        } catch (e) {
            console.log(`  Blender validation skipped: ${e.message}`);
        }
    }

    console.log('\n=== Pipeline Complete ===');
    return {
        prompt: userPrompt,
        expandedPrompt: flux2Prompt,
        segmentationPrompt,
        generatedImage,
        outputs: outputFiles
    };
}

// CLI interface
async function main() {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        console.log('Usage: text-to-3d-pipeline.js "<prompt>" [--portrait] [--no-blender]');
        console.log('\nExample:');
        console.log('  text-to-3d-pipeline.js "vintage sports car"');
        console.log('  text-to-3d-pipeline.js "tall fantasy tower" --portrait');
        process.exit(1);
    }

    const prompt = args[0];
    const options = {
        orientation: args.includes('--portrait') ? 'portrait' : 'landscape',
        skipBlenderValidation: args.includes('--no-blender')
    };

    try {
        const result = await runPipeline(prompt, options);
        console.log('\nResult:', JSON.stringify(result, null, 2));
    } catch (error) {
        console.error('\nPipeline failed:', error.message);
        process.exit(1);
    }
}

// Export for programmatic use
module.exports = { runPipeline, expandPromptForFlux2, distillForSegmentation };

// Run if called directly
if (require.main === module) {
    main();
}
