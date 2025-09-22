# Reference: MCP Tools API

This document provides a detailed reference for each Model Context Protocol (MCP) tool available in the environment.

### Tool Types

-   **Direct**: The tool is a self-contained executable that runs directly within the `multi-agent-container`.
-   **Bridge**: The tool is a lightweight client that forwards requests to a service running in the `gui-tools-container`.

---

### `imagemagick-mcp`

-   **Type**: Direct
-   **Source**: `core-assets/mcp-tools/imagemagick_mcp.py`
-   **Purpose**: Image creation and manipulation using the ImageMagick `convert` tool.

#### Methods

**`create`**
Creates a new, simple image.

-   **Parameters**:
    -   `width` (int): The image width. Default: `100`.
    -   `height` (int): The image height. Default: `100`.
    -   `color` (string): The background color (e.g., "red", "#FF0000"). Default: `"white"`.
    -   `output` (string): **Required**. The path to save the output image.
-   **Example**:
    ```json
    {
      "method": "create",
      "params": {
        "width": 100, "height": 100, "color": "red", "output": "red_square.png"
      }
    }
    ```

**`convert`**
Runs a generic ImageMagick `convert` command.

-   **Parameters**:
    -   `args` (list[string]): **Required**. A list of arguments to pass to the `convert` command.
-   **Example** (resize an image):
    ```json
    {
      "method": "convert",
      "params": {
        "args": ["input.jpg", "-resize", "50%", "output.jpg"]
      }
    }
    ```

**`resize`**
Resizes an image to specified dimensions.

-   **Parameters**:
    -   `input` (string): **Required**. Path to the input image.
    -   `output` (string): **Required**. Path to save the resized image.
    -   `width` (int): Target width in pixels.
    -   `height` (int): Target height in pixels.
    -   `maintain_aspect_ratio` (bool): Keep aspect ratio. Default: `true`.
-   **Example**:
    ```json
    {
      "method": "resize",
      "params": {
        "input": "photo.jpg",
        "output": "thumbnail.jpg",
        "width": 150,
        "height": 150
      }
    }
    ```

---

### `blender-mcp`

-   **Type**: Bridge
-   **Source**: `core-assets/scripts/mcp-blender-client.js`
-   **Server**: `gui-based-tools-docker/addon.py`
-   **Purpose**: 3D modeling, rendering, and scene manipulation via an external Blender instance.

#### Methods

The bridge forwards any `tool` and `params` to the Blender server. The available methods are defined in `addon.py`.

**`execute_code`**
Executes arbitrary Python code within Blender's context.

-   **Parameters**:
    -   `code` (string): **Required**. The Python code to execute.
-   **Example**:
    ```json
    {
      "tool": "execute_code",
      "params": {
        "code": "import bpy; bpy.ops.mesh.primitive_cube_add(size=2)"
      }
    }
    ```

**`get_viewport_screenshot`**
Captures the current 3D viewport as an image.

-   **Parameters**:
    -   `filepath` (string): **Required**. The path (inside the container) to save the screenshot.
-   **Example**:
    ```json
    {
      "tool": "get_viewport_screenshot",
      "params": { "filepath": "/workspace/blender_view.png" }
    }
    ```

**`import_model`**
Imports a 3D model file into the scene.

-   **Parameters**:
    -   `filepath` (string): **Required**. Path to the model file.
    -   `format` (string): File format (e.g., "obj", "fbx", "gltf"). Default: auto-detected.
-   **Example**:
    ```json
    {
      "tool": "import_model",
      "params": {
        "filepath": "/workspace/model.obj",
        "format": "obj"
      }
    }
    ```

**`export_scene`**
Exports the current scene to a file.

-   **Parameters**:
    -   `filepath` (string): **Required**. Path to save the exported file.
    -   `format` (string): Export format (e.g., "gltf", "fbx", "obj").
    -   `selected_only` (bool): Export only selected objects. Default: `false`.
-   **Example**:
    ```json
    {
      "tool": "export_scene",
      "params": {
        "filepath": "/workspace/scene.gltf",
        "format": "gltf"
      }
    }
    ```

---

### `pbr-generator-mcp`

-   **Type**: Bridge
-   **Source**: `core-assets/mcp-tools/pbr_mcp_client.py`
-   **Server**: `gui-based-tools-docker/tessellating-pbr-generator/pbr_mcp_server.py`
-   **Purpose**: Generates Physically Based Rendering (PBR) texture maps.

#### Methods

**`generate_material`**
Generates a set of PBR textures.

-   **Parameters**:
    -   `material` (string): **Required**. The name of the material to generate (e.g., "wood", "metal", "stone", "fabric").
    -   `resolution` (string): The output resolution (e.g., "512x512", "1024x1024", "2048x2048"). Default: "1024x1024".
    -   `types` (list[string]): A list of texture types to generate. Default: `["diffuse", "normal", "roughness", "metallic", "displacement"]`.
    -   `output` (string): **Required**. The directory path to save the generated textures.
-   **Example**:
    ```json
    {
      "tool": "generate_material",
      "params": {
        "material": "fabric",
        "resolution": "1024x1024",
        "types": ["diffuse", "normal"],
        "output": "/workspace/textures/fabric"
      }
    }
    ```

**`list_materials`**
Lists all available material presets.

-   **Parameters**: None
-   **Example**:
    ```json
    {
      "tool": "list_materials",
      "params": {}
    }
    ```

---

### `kicad-mcp`

-   **Type**: Direct
-   **Source**: `core-assets/mcp-tools/kicad_mcp.py`
-   **Purpose**: Interacts with `kicad-cli` for Electronic Design Automation (EDA) tasks.

#### Methods

**`create_project`**
Creates a new, empty KiCad project.

-   **Parameters**:
    -   `project_name` (string): **Required**. The name for the new project.
    -   `project_dir` (string): The directory to create the project in. Default: `/workspace`.
-   **Example**:
    ```json
    {
      "method": "create_project",
      "params": { "project_name": "my_first_pcb" }
    }
    ```

**`export_gerbers`**
Exports Gerber files from a `.kicad_pcb` file.

-   **Parameters**:
    -   `pcb_file` (string): **Required**. The path to the KiCad PCB file.
    -   `output_dir` (string): The directory to save the Gerber files. Default: `/workspace/gerbers`.
-   **Example**:
    ```json
    {
      "method": "export_gerbers",
      "params": { "pcb_file": "my_first_pcb/my_first_pcb.kicad_pcb" }
    }
    ```

**`export_svg`**
Exports the PCB as an SVG image.

-   **Parameters**:
    -   `pcb_file` (string): **Required**. Path to the KiCad PCB file.
    -   `output_file` (string): **Required**. Path to save the SVG file.
    -   `layers` (list[string]): Layers to export. Default: all layers.
-   **Example**:
    ```json
    {
      "method": "export_svg",
      "params": {
        "pcb_file": "board.kicad_pcb",
        "output_file": "board_preview.svg"
      }
    }
    ```

---

### `ngspice-mcp`

-   **Type**: Direct
-   **Source**: `core-assets/mcp-tools/ngspice_mcp.py`
-   **Purpose**: Runs circuit simulations using NGSpice.

#### Methods

**`run_simulation`**
Executes a simulation from a SPICE netlist.

-   **Parameters**:
    -   `netlist` (string): **Required**. A string containing the full SPICE netlist to simulate.
    -   `analysis_type` (string): Type of analysis ("tran", "ac", "dc"). Default: auto-detected from netlist.
-   **Example**:
    ```json
    {
      "method": "run_simulation",
      "params": {
        "netlist": "V1 1 0 1\nR1 1 0 1k\n.tran 1u 1m\n.end"
      }
    }
    ```

**`load_netlist`**
Loads and validates a netlist from a file.

-   **Parameters**:
    -   `filepath` (string): **Required**. Path to the netlist file.
-   **Example**:
    ```json
    {
      "method": "load_netlist",
      "params": { "filepath": "/workspace/circuits/amplifier.cir" }
    }
    ```

---

### `qgis-mcp`

-   **Type**: Bridge
-   **Source**: `core-assets/mcp-tools/qgis_mcp.py`
-   **Server**: QGIS MCP Plugin
-   **Purpose**: Geospatial analysis and map generation via an external QGIS instance.

#### Methods

The bridge forwards any `tool` and `params` to the QGIS server. Common methods include:

**`get_qgis_version`**
Retrieves the version of the running QGIS instance.

-   **Parameters**: None
-   **Example**:
    ```json
    {
      "tool": "get_qgis_version",
      "params": {}
    }
    ```

**`load_vector`**
Loads a vector data file (shapefile, GeoJSON, etc.).

-   **Parameters**:
    -   `filepath` (string): **Required**. Path to the vector file.
    -   `name` (string): Layer name in QGIS. Default: filename.
-   **Example**:
    ```json
    {
      "tool": "load_vector",
      "params": {
        "filepath": "/workspace/boundaries.shp",
        "name": "City Boundaries"
      }
    }
    ```

**`buffer`**
Creates a buffer around features in a layer.

-   **Parameters**:
    -   `layer` (string): **Required**. Name of the layer to buffer.
    -   `distance` (float): **Required**. Buffer distance.
    -   `units` (string): Distance units ("meters", "feet", "degrees"). Default: "meters".
    -   `output_name` (string): Name for the output layer. Default: "{layer}_buffered".
-   **Example**:
    ```json
    {
      "tool": "buffer",
      "params": {
        "layer": "City Boundaries",
        "distance": 1000,
        "units": "meters"
      }
    }
    ```

**`export_map`**
Exports the current map view as an image.

-   **Parameters**:
    -   `filepath` (string): **Required**. Path to save the map image.
    -   `format` (string): Image format ("png", "jpg", "pdf"). Default: "png".
    -   `dpi` (int): Resolution in DPI. Default: 300.
    -   `width` (int): Image width in pixels. Default: 1920.
    -   `height` (int): Image height in pixels. Default: 1080.
-   **Example**:
    ```json
    {
      "tool": "export_map",
      "params": {
        "filepath": "/workspace/city_map.png",
        "format": "png",
        "dpi": 300
      }
    }
    ```

---

## Error Handling

All tools follow a consistent error response format:

```json
{
  "error": "Error message describing what went wrong",
  "code": "ERROR_CODE",
  "details": {
    "additional": "context"
  }
}
```

Common error codes:
- `INVALID_PARAMS`: Missing or invalid parameters
- `FILE_NOT_FOUND`: Referenced file doesn't exist
- `CONNECTION_ERROR`: Bridge tool can't connect to server
- `EXECUTION_ERROR`: Tool execution failed
- `TIMEOUT`: Operation timed out

## Best Practices

1. **Always validate file paths**: Ensure files exist before referencing them
2. **Use absolute paths**: All file paths should be absolute, starting with `/workspace`
3. **Check tool availability**: Use `mcp-helper.sh list-tools` to verify tool availability
4. **Handle errors gracefully**: Always check for error responses and handle them appropriately
5. **Batch operations**: When possible, batch multiple operations to reduce overhead