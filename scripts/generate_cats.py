#!/usr/bin/env python3
"""Generate 4 highly differentiated cat images using FLUX model in ComfyUI."""

import requests
import json
import time
import base64
import random
from pathlib import Path

COMFYUI_URL = "http://localhost:8188"
OUTPUT_DIR = Path("/home/devuser/workspace/project/output/cats")

# Four highly differentiated cat prompts
CAT_PROMPTS = [
    {
        "name": "cyberpunk_neon_cat",
        "prompt": "a futuristic cyberpunk cat with glowing neon cyan and magenta eyes, circuit patterns on fur, holographic collar, metallic whiskers, standing in a neon-lit rain-soaked alley, ultra detailed, 8k, cinematic lighting, dramatic shadows",
        "width": 1024,
        "height": 1024,
    },
    {
        "name": "watercolor_garden_cat",
        "prompt": "a soft watercolor painting of an elegant Persian cat sitting in a blooming spring garden, delicate brushstrokes, pastel pink and purple flowers, dreamy atmosphere, artistic, flowing colors, gentle lighting, impressionist style",
        "width": 1024,
        "height": 1024,
    },
    {
        "name": "glass_sculpture_cat",
        "prompt": "a translucent 3D rendered glass sculpture of a sitting cat, crystal clear material with rainbow light refraction, smooth curves, reflective surfaces, studio lighting, white background, photorealistic rendering, elegant and minimalist",
        "width": 1024,
        "height": 1024,
    },
    {
        "name": "noir_detective_cat",
        "prompt": "a vintage film noir style black and white photograph of a sophisticated cat detective wearing a fedora hat and trench coat, sitting at a desk with dramatic venetian blind shadows, cigarette smoke, 1940s atmosphere, high contrast, moody lighting",
        "width": 1024,
        "height": 1024,
    }
]


def create_flux_workflow(prompt, width=1024, height=1024, seed=None):
    """Create a FLUX workflow JSON for ComfyUI."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    workflow = {
        "6": {
            "inputs": {
                "text": prompt,
                "clip": ["30", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Positive)"}
        },
        "8": {
            "inputs": {
                "samples": ["31", 0],
                "vae": ["30", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        },
        "27": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            },
            "class_type": "EmptySD3LatentImage"
        },
        "30": {
            "inputs": {
                "ckpt_name": "flux1-schnell-fp8.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "31": {
            "inputs": {
                "seed": seed,
                "steps": 4,  # Schnell is optimized for 4 steps
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["30", 0],
                "positive": ["35", 0],
                "negative": ["33", 0],
                "latent_image": ["27", 0]
            },
            "class_type": "KSampler"
        },
        "33": {
            "inputs": {
                "text": "blurry, low quality, distorted, ugly, bad anatomy",
                "clip": ["30", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative)"}
        },
        "35": {
            "inputs": {
                "guidance": 3.5,
                "conditioning": ["6", 0]
            },
            "class_type": "FluxGuidance"
        }
    }

    return workflow


def queue_prompt(workflow):
    """Queue a prompt to ComfyUI."""
    payload = {"prompt": workflow}
    response = requests.post(f"{COMFYUI_URL}/prompt", json=payload)
    return response.json()


def get_history(prompt_id):
    """Get the history for a prompt ID."""
    response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    return response.json()


def get_image(filename, subfolder, folder_type):
    """Download an image from ComfyUI."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{COMFYUI_URL}/view", params=data)
    return response.content


def wait_for_completion(prompt_id, timeout=600, poll_interval=2):
    """Wait for a prompt to complete."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        history = get_history(prompt_id)

        if prompt_id in history:
            prompt_data = history[prompt_id]
            if "outputs" in prompt_data:
                return prompt_data

        time.sleep(poll_interval)
        print(".", end="", flush=True)

    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def generate_cat_image(cat_config):
    """Generate a single cat image."""
    print(f"\n{'='*60}")
    print(f"Generating: {cat_config['name']}")
    print(f"Prompt: {cat_config['prompt'][:80]}...")
    print(f"{'='*60}")

    # Create workflow
    workflow = create_flux_workflow(
        cat_config["prompt"],
        cat_config["width"],
        cat_config["height"]
    )

    # Queue the prompt
    print("Queueing prompt...")
    result = queue_prompt(workflow)
    prompt_id = result.get("prompt_id")

    if not prompt_id:
        print(f"Error: {result}")
        return None

    print(f"Prompt ID: {prompt_id}")
    print("Waiting for completion", end="")

    # Wait for completion
    try:
        prompt_data = wait_for_completion(prompt_id, timeout=600)
        print("\nGeneration complete!")
    except TimeoutError as e:
        print(f"\nError: {e}")
        return None

    # Get the image
    outputs = prompt_data.get("outputs", {})

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for image_data in node_output["images"]:
                filename = image_data["filename"]
                subfolder = image_data.get("subfolder", "")
                folder_type = image_data.get("type", "output")

                print(f"Downloading: {filename}")
                image_bytes = get_image(filename, subfolder, folder_type)

                # Save the image
                output_path = OUTPUT_DIR / f"{cat_config['name']}.png"
                output_path.write_bytes(image_bytes)
                print(f"Saved to: {output_path}")

                return str(output_path)

    print("No image found in output")
    return None


def main():
    """Generate all 4 cat images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("FLUX CAT IMAGE GENERATOR")
    print("Generating 4 highly differentiated cat images")
    print("="*60)

    # Check if ComfyUI is running
    try:
        response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        print(f"\n✓ ComfyUI is running (v{response.json()['system']['comfyui_version']})")
    except Exception as e:
        print(f"\n✗ Error: ComfyUI is not responding at {COMFYUI_URL}")
        print(f"  Details: {e}")
        return

    # Generate each cat image
    results = []
    for i, cat_config in enumerate(CAT_PROMPTS, 1):
        print(f"\n[{i}/{len(CAT_PROMPTS)}]", end=" ")
        output_path = generate_cat_image(cat_config)
        results.append({
            "name": cat_config["name"],
            "path": output_path,
            "prompt": cat_config["prompt"]
        })

        # Brief pause between generations
        if i < len(CAT_PROMPTS):
            time.sleep(2)

    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)

    for i, result in enumerate(results, 1):
        status = "✓" if result["path"] else "✗"
        print(f"{status} [{i}] {result['name']}")
        if result["path"]:
            print(f"    Path: {result['path']}")

    print(f"\nAll images saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
