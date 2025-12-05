---
name: ComfyUI
description: AI image and video generation using ComfyUI node-based workflow system with Stable Diffusion, FLUX, and other diffusion models via local or distributed GPU compute (Salad Cloud)
---

# ComfyUI Skill

This skill enables Claude to interact with ComfyUI for AI image/video generation, workflow management, and distributed GPU compute via Salad Cloud API.

## Capabilities

- Generate images using text prompts (text2img)
- Generate images from images (img2img)
- Create and execute node-based workflows
- Deploy to Salad Cloud for distributed GPU compute
- Manage models, LoRAs, and checkpoints
- Video generation (AnimateDiff, CogVideoX, HunyuanVideo)
- Upscaling and post-processing

## When to Use This Skill

Use this skill when you need to:
- Generate AI images from text descriptions
- Create image-to-image transformations
- Design ComfyUI workflows programmatically
- Deploy image generation at scale on distributed GPUs
- Batch process image generation tasks
- Fine-tune or use LoRA models with FLUX/SD
- Generate AI videos from text or images

## Prerequisites

### Local Setup
- ComfyUI installed: `/home/devuser/ComfyUI/`
- Python venv: `source /home/devuser/ComfyUI/venv/bin/activate`
- GPU with CUDA support (or --cpu flag for testing)
- Default port: 8188

### Distributed (Salad Cloud)
- SALAD_API_KEY environment variable
- SALAD_ORG_NAME environment variable
- salad-cloud-sdk: `pip install salad-cloud-sdk`

## Instructions

### Local ComfyUI Operations

#### Start ComfyUI Server
```bash
cd /home/devuser/ComfyUI
source venv/bin/activate
python main.py --listen 0.0.0.0 --port 8188
```

#### Start with GPU
```bash
python main.py --listen 0.0.0.0 --port 8188
```

#### Start in CPU mode (testing)
```bash
python main.py --listen 0.0.0.0 --port 8188 --cpu
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8188/health
```

#### Ready Check
```bash
curl http://localhost:8188/ready
```

#### List Available Models
```bash
curl http://localhost:8188/models
```

#### Text to Image (Simple)
```bash
curl -X POST "http://localhost:8188/workflow/text2img" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A beautiful sunset over the ocean",
      "width": 1024,
      "height": 1024,
      "steps": 20,
      "cfg_scale": 7.5
    }
  }' | jq -r '.images[0]' | base64 -d > image.png
```

#### Submit Raw ComfyUI Prompt
```bash
curl -X POST "http://localhost:8188/prompt" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": {
      "6": {
        "inputs": { "text": "your prompt here", "clip": ["30", 1] },
        "class_type": "CLIPTextEncode"
      },
      ...
    }
  }'
```

### Workflow Parameters

#### text2img Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| prompt | string | required | - | Positive prompt for image generation |
| width | integer | 1024 | 256-2048 | Image width in pixels |
| height | integer | 1024 | 256-2048 | Image height in pixels |
| seed | integer | random | - | Seed for reproducibility |
| steps | integer | 20 | 1-100 | Number of sampling steps |
| cfg_scale | number | 1.0 | 0-20 | Classifier-free guidance scale |
| sampler_name | string | "euler" | see list | Sampling algorithm |
| scheduler | string | "simple" | see list | Noise scheduler |
| denoise | number | 1.0 | 0-1 | Denoising strength |
| guidance | number | 3.5 | 0-10 | FLUX guidance scale |

#### Available Samplers
euler, euler_cfg_pp, euler_ancestral, euler_ancestral_cfg_pp, heun, heunpp2, dpm_2, dpm_2_ancestral, lms, dpm_fast, dpm_adaptive, dpmpp_2s_ancestral, dpmpp_2s_ancestral_cfg_pp, dpmpp_sde, dpmpp_sde_gpu, dpmpp_2m, dpmpp_2m_cfg_pp, dpmpp_2m_sde, dpmpp_2m_sde_gpu, dpmpp_3m_sde, dpmpp_3m_sde_gpu, ddpm, lcm, ipndm, ipndm_v, deis, ddim, uni_pc, uni_pc_bh2

#### Available Schedulers
normal, karras, exponential, sgm_uniform, simple, ddim_uniform, beta, linear_quadratic

### Output Conversion

Convert output to JPEG or WebP:
```json
{
  "convert_output": {
    "format": "webp",
    "options": {
      "quality": 85,
      "lossless": false
    }
  }
}
```

### Webhook Support

Receive completed images via webhook:
```json
{
  "webhook": "https://your-server.com/webhook",
  "input": { "prompt": "..." }
}
```

## Salad Cloud Deployment

### Initialize Salad SDK
```python
import os
from salad_cloud_sdk import SaladCloudSdk

sdk = SaladCloudSdk(api_key=os.environ['SALAD_API_KEY'])
org_name = os.environ.get('SALAD_ORG_NAME', 'default-org')
```

### List GPU Classes
```python
gpu_classes = sdk.organization_data.list_gpu_classes(organization_name=org_name)
for gpu in gpu_classes.items:
    print(f"{gpu.name}: {gpu.display_name}")
```

### Create ComfyUI Container Group
```python
from salad_cloud_sdk.models import (
    CreateContainerGroup,
    ContainerGroupPriority,
    ContainerResourceRequirements,
    CountryCode,
)

container_group = CreateContainerGroup(
    name="comfyui-worker",
    display_name="ComfyUI Worker",
    container=ContainerResourceRequirements(
        image="ghcr.io/comfyanonymous/comfyui:latest",
        resources={
            "cpu": 4,
            "memory": 30720,  # 30GB RAM recommended
            "gpu_classes": ["rtx_4090", "rtx_3090", "a100"]
        },
        environment_variables={
            "COMFYUI_LISTEN": "0.0.0.0",
            "COMFYUI_PORT": "8188"
        }
    ),
    replicas=3,  # Minimum 3 for production
    priority=ContainerGroupPriority.MEDIUM,
    country_codes=[CountryCode.US, CountryCode.CA, CountryCode.GB],
    networking={
        "protocol": "http",
        "port": 8188,
        "auth": False
    }
)

result = sdk.container_groups.create_container_group(
    organization_name=org_name,
    project_name="default",
    request_body=container_group
)
```

### Get Quotas
```python
quotas = sdk.quotas.get_quotas(organization_name=org_name)
print(f"Max container groups: {quotas.container_groups_quotas.max_created_container_groups}")
```

### List Inference Endpoints
```python
endpoints = sdk.inference_endpoints.list_inference_endpoints(
    organization_name=org_name,
    project_name="default"
)
```

## ComfyUI Workflow JSON Structure

### Basic FLUX Workflow
```json
{
  "6": {
    "inputs": {
      "text": "your prompt here",
      "clip": ["30", 1]
    },
    "class_type": "CLIPTextEncode",
    "_meta": { "title": "CLIP Text Encode (Positive Prompt)" }
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
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptySD3LatentImage"
  },
  "30": {
    "inputs": {
      "ckpt_name": "flux1-dev-fp8.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "31": {
    "inputs": {
      "seed": 793373912447585,
      "steps": 20,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1,
      "model": ["30", 0],
      "positive": ["35", 0],
      "negative": ["33", 0],
      "latent_image": ["27", 0]
    },
    "class_type": "KSampler"
  },
  "33": {
    "inputs": {
      "text": "",
      "clip": ["30", 1]
    },
    "class_type": "CLIPTextEncode"
  },
  "35": {
    "inputs": {
      "guidance": 3.5,
      "conditioning": ["6", 0]
    },
    "class_type": "FluxGuidance"
  }
}
```

## Supported Models

### Image Generation
- FLUX.1-Dev (FP8) - High quality, text generation, non-commercial
- FLUX.1-Schnell (FP8) - Fast generation
- Stable Diffusion 3.5 Large/Medium
- SDXL with Refiner
- DreamShaper 8

### Video Generation
- AnimateDiff
- CogVideoX-2B
- HunyuanVideo (FP16)
- LTX-Video
- Mochi Video (FP8)
- Cosmos 1.0 (Text2World)
- WAN 2.1 (I2V 720p)

## Examples

### Example 1: Generate FLUX Image
```bash
curl -X POST "http://localhost:8188/workflow/text2img" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A majestic dragon perched on a crystal mountain, cinematic lighting, 8k resolution",
      "width": 1024,
      "height": 768,
      "steps": 25,
      "guidance": 3.5,
      "sampler_name": "euler",
      "scheduler": "simple"
    }
  }' | jq -r '.images[0]' | base64 -d > dragon.png
```

### Example 2: Batch Generation with Python
```python
import requests
import base64

def generate_image(prompt, output_path, **kwargs):
    response = requests.post(
        "http://localhost:8188/workflow/text2img",
        json={
            "input": {
                "prompt": prompt,
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024),
                "steps": kwargs.get("steps", 20),
                "seed": kwargs.get("seed", -1),
            }
        }
    )
    data = response.json()
    if "images" in data:
        image_data = base64.b64decode(data["images"][0])
        with open(output_path, "wb") as f:
            f.write(image_data)
        return True
    return False

# Generate multiple images
prompts = [
    "A serene Japanese garden at sunset",
    "Cyberpunk cityscape with neon lights",
    "Portrait of an astronaut on Mars"
]

for i, prompt in enumerate(prompts):
    generate_image(prompt, f"output_{i}.png")
```

### Example 3: Salad Cloud Production Deployment
```python
import os
from salad_cloud_sdk import SaladCloudSdk

sdk = SaladCloudSdk(api_key=os.environ['SALAD_API_KEY'])

# Deploy with recommended production settings
container_config = {
    "name": "flux-production",
    "replicas": 5,  # Over-provision for reliability
    "resources": {
        "cpu": 4,
        "memory": 30720,
        "gpu_classes": ["rtx_4090"]  # 24GB VRAM recommended
    }
}
```

## Hardware Recommendations

| Model | VRAM | System RAM | Notes |
|-------|------|------------|-------|
| FLUX.1-Dev FP8 | 16GB+ | 30GB | RTX 4090 recommended |
| FLUX.1-Schnell | 12GB+ | 24GB | Faster inference |
| SD 3.5 Large | 16GB+ | 24GB | High quality |
| SDXL | 12GB+ | 16GB | Good balance |
| AnimateDiff | 16GB+ | 32GB | Video generation |

## Error Handling

Common errors and solutions:
- **CUDA out of memory**: Reduce resolution or batch size
- **Model not found**: Check checkpoint path in models directory
- **Connection refused**: Ensure ComfyUI server is running
- **Timeout**: Increase timeout for large generations

## Integration with Other Skills

Works well with:
- `imagemagick` skill for image post-processing
- `ffmpeg-processing` skill for video processing
- `blender` skill for 3D-to-2D workflows
- `pytorch-ml` skill for custom model training

## Performance Notes

- Image generation (1024x1024): 3-15 seconds on RTX 4090
- Video generation: varies by length and model
- Distributed compute: account for network latency
- Use webhooks for async operations in production

## Files and Directories

```
/home/devuser/ComfyUI/
  venv/           # Python virtual environment
  models/         # Model checkpoints
    checkpoints/  # Main models
    loras/        # LoRA adapters
    vae/          # VAE models
  custom_nodes/   # Custom node packages
  input/          # Input images
  output/         # Generated outputs
  scripts/        # Utility scripts
    test_salad_api.py  # Salad SDK test
```

## Salad Recipes Reference

All recipes available at `/home/devuser/salad-recipes/src/`:

### Image Generation Recipes
| Recipe | Model | Workflow | Container Config |
|--------|-------|----------|------------------|
| flux1-dev-fp8-comfyui | FLUX.1-Dev FP8 | workflow.json | container-group.json |
| flux1-schnell-fp8-comfyui | FLUX.1-Schnell FP8 | workflow.json | container-group.json |
| flux1-dev-lora-comfyui | FLUX.1-Dev + LoRA | workflow.json | container-group.json |
| sd3.5-large-comfyui | SD 3.5 Large | workflow.json | container-group.json |
| sd3.5-medium-comfyui | SD 3.5 Medium | workflow.json | container-group.json |
| sdxl-with-refiner-comfyui | SDXL + Refiner | workflow.json | container-group.json |
| dreamshaper8-comfyui | DreamShaper 8 | workflow.json | container-group.json |

### Video Generation Recipes
| Recipe | Model | Workflow | Notes |
|--------|-------|----------|-------|
| animatediff-comfyui | AnimateDiff | workflow.json | Animation from images |
| cogvideox-2b-comfyui | CogVideoX 2B | - | Text-to-video |
| hunyuanvideo-fp16-comfyui | HunyuanVideo FP16 | - | High quality video |
| ltx-video-2b-v0.9.1-comfyui | LTX-Video 2B | workflow.json | Fast video generation |
| mochi-video-fp8-comfyui | Mochi Video FP8 | - | Efficient video |
| cosmos1.0-7b-text2world-comfyui | Cosmos Text2World | workflow.json | World generation |
| wan2.1-i2v-720p-comfyui | WAN 2.1 I2V | prompt.json | Image-to-video 720p |

### LLM Recipes (Text Generation Inference)
| Recipe | Model | Container Config |
|--------|-------|------------------|
| tgi-llama-3.1-8b-instruct | Llama 3.1 8B | container-group.json |
| tgi-llama-3.2-11b-vision-instruct | Llama 3.2 Vision 11B | container-group.json |
| tgi-mistral-7b | Mistral 7B | container-group.json |
| tgi-nemo-12b-instruct-fp8 | Nemo 12B FP8 | container-group.json |
| tgi-qwen2.5-vl-3b-instruct | Qwen 2.5 VL 3B | container-group.json |
| tgi-qwen2.5-vl-7b-instruct | Qwen 2.5 VL 7B | container-group.json |
| tgi-qwen3-8b | Qwen 3 8B | container-group.json |
| tgi-lyra-12b-darkness | Lyra 12B | container-group.json |

### Other Recipes
| Recipe | Purpose |
|--------|---------|
| yolov8 | Object detection (OpenAPI available) |
| ollama | Local LLM server |
| ollama-llama3.1 | Ollama with Llama 3.1 |
| ubuntu-dev | Development environment |
| hello-world | Template example |
| sogni-flux-worker | Sogni FLUX worker |
| sogni-stable-diffusion-worker | Sogni SD worker |

### Loading Recipe Workflows

```python
import json

# Load a workflow
with open('/home/devuser/salad-recipes/src/flux1-dev-fp8-comfyui/workflow.json') as f:
    workflow = json.load(f)

# Load container group config for Salad deployment
with open('/home/devuser/salad-recipes/src/flux1-dev-fp8-comfyui/container-group.json') as f:
    container_config = json.load(f)

# Load OpenAPI spec (where available)
with open('/home/devuser/salad-recipes/src/flux1-dev-fp8-comfyui/openapi.json') as f:
    api_spec = json.load(f)
```

### Benchmark Data Available

Performance benchmarks in `benchmark/` subdirectories:
- `flux1-dev-fp8-comfyui/benchmark/4090.json` - RTX 4090 benchmarks
- `sd3.5-medium-comfyui/benchmark/` - RTX 3090/4090 comparisons
- `ltx-video-2b-v0.9.1-comfyui/benchmark/` - Video generation benchmarks

## MCP Server Tools

The ComfyUI skill includes an MCP server at `mcp-server/` that provides autonomous workflow tools:

### Available Tools

| Tool | Description |
|------|-------------|
| `workflow_submit` | Submit ComfyUI workflow JSON for execution |
| `workflow_status` | Check job status by ID |
| `workflow_cancel` | Cancel a running job |
| `model_list` | List available models (checkpoints, loras, vae) |
| `image_generate` | Convenience text2img with FLUX |
| `video_generate` | Video generation (AnimateDiff) |
| `display_capture` | Screenshot display :1 for visual feedback |
| `output_list` | List generated outputs |
| `chat_workflow` | Natural language to workflow via LLM |

### MCP Server Setup

```bash
cd /home/devuser/.claude/skills/comfyui/mcp-server
npm install
npm start
```

### MCP Configuration

Add to your MCP configuration:
```json
{
  "comfyui": {
    "command": "node",
    "args": ["/home/devuser/.claude/skills/comfyui/mcp-server/server.js"],
    "env": {
      "COMFYUI_URL": "http://localhost:8188"
    }
  }
}
```

### Visual Display Monitoring

The skill can capture ComfyUI's UI on display :1 using Playwright for visual feedback to Claude. Use `display_capture` tool to get screenshots.

### LLM Workflow Generation

Use `chat_workflow` to generate ComfyUI workflows from natural language:
```
chat_workflow: "Create an anime portrait with blue hair and golden eyes"
```

This uses the Z.AI service (port 9600) or Anthropic API to generate valid workflow JSON.

## References

- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- ComfyUI API: https://github.com/SaladTechnologies/comfyui-api
- Salad Recipes: https://github.com/SaladTechnologies/salad-recipes
- Salad Cloud SDK: https://portal.salad.com
- Local Recipes: /home/devuser/salad-recipes/src/
