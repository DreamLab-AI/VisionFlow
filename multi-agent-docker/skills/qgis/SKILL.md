---
name: "QGIS Geographic"
description: "Control QGIS for geographic information system operations, layer management, and spatial analysis. Use when working with GIS data, creating maps, analyzing geospatial information, processing vector/raster layers, or performing spatial queries."
---

# QGIS Geographic Skill

## What This Skill Does

Enables Claude to interact with QGIS Desktop for comprehensive GIS operations including project management, layer manipulation, spatial processing, and map production through socket-based communication on port 2801.

## Prerequisites

- QGIS 3.X (LTS recommended: 3.28+ or 3.34+)
- QGIS Python API enabled
- Socket server plugin active on port 2801
- Python 3.9+ with PyQGIS bindings

## When to Use This Skill

Use this skill when you need to:
- Load and manage geospatial data (shapefiles, GeoJSON, GeoTIFF)
- Perform spatial analysis and geoprocessing
- Create cartographic maps with styling
- Execute QGIS processing algorithms
- Automate GIS workflows
- Generate spatial reports and visualizations

---

## Quick Start

### Basic Project Setup

```bash
# Load an existing QGIS project
Use QGIS skill to load project from /path/to/project.qgz

# Or create a new project
Create a new QGIS project with WGS84 CRS (EPSG:4326)
```

### Add Your First Layer

```bash
# Add vector layer (shapefile, GeoJSON, etc.)
Add vector layer from /data/cities.shp to current QGIS project

# Add raster layer (GeoTIFF, etc.)
Add raster layer from /data/elevation.tif with hillshade styling
```

### Run Simple Analysis

```bash
# Buffer analysis
Create 1km buffer around cities layer and save to /output/buffers.shp

# Intersection
Intersect roads layer with parcels layer
```

---

## Step-by-Step Guide

### 1. Project Management

#### Create New Project

```python
# Via QGIS skill
create_project({
  "name": "MyGISProject",
  "crs": "EPSG:4326",  # WGS84
  "save_path": "/projects/mygis.qgz"
})
```

#### Load Existing Project

```python
load_project({
  "path": "/projects/existing.qgz"
})
```

### 2. Adding Layers

#### Vector Layers

Supported formats: Shapefile (.shp), GeoJSON (.json,.geojson), GeoPackage (.gpkg), KML, CSV with geometry

```python
add_vector_layer({
  "path": "/data/cities.shp",
  "name": "Cities",           # Optional: custom layer name
  "style": "categorized"      # Optional: default, categorized, graduated
})
```

#### Raster Layers

Supported formats: GeoTIFF (.tif,.tiff), JPEG2000, ECW, MrSID, ASCII Grid

```python
add_raster_layer({
  "path": "/data/elevation.tif",
  "name": "Elevation",
  "colormap": "terrain"       # Optional: grayscale, terrain, rainbow
})
```

### 3. Spatial Processing

#### Buffer Analysis

```python
run_processing({
  "algorithm": "native:buffer",
  "parameters": {
    "INPUT": "Cities",
    "DISTANCE": 5000,         # meters
    "SEGMENTS": 10,
    "OUTPUT": "/output/city_buffers.shp"
  }
})
```

#### Intersection

```python
run_processing({
  "algorithm": "native:intersection",
  "parameters": {
    "INPUT": "Parcels",
    "OVERLAY": "Flood_Zones",
    "OUTPUT": "/output/parcels_in_flood.shp"
  }
})
```

#### Clip

```python
run_processing({
  "algorithm": "native:clip",
  "parameters": {
    "INPUT": "Roads",
    "OVERLAY": "Study_Area",
    "OUTPUT": "/output/roads_clipped.shp"
  }
})
```

### 4. Styling and Visualization

#### Apply Symbology

```python
apply_style({
  "layer": "Cities",
  "type": "categorized",
  "field": "population",
  "categories": [
    {"value": "< 100000", "color": "#fee5d9"},
    {"value": "100000-500000", "color": "#fc9272"},
    {"value": "> 500000", "color": "#de2d26"}
  ]
})
```

#### Create Map Layout

```python
create_layout({
  "name": "Population Map",
  "size": "A4",
  "orientation": "landscape",
  "elements": [
    {"type": "map", "extent": "current"},
    {"type": "legend", "position": "bottom-right"},
    {"type": "scale", "position": "bottom-left"},
    {"type": "title", "text": "City Population Distribution"}
  ]
})
```

### 5. Export Maps

```python
export_map({
  "output_path": "/output/population_map.png",
  "resolution": 300,          # DPI
  "width": 1920,
  "height": 1080
})
```

---

## Tool Functions

### `create_project`
Create a new QGIS project.

Parameters:
- `name` (optional): Project name
- `crs` (optional): Coordinate reference system (default: "EPSG:4326")
- `save_path` (optional): Path to save project file

### `load_project`
Load an existing QGIS project file.

Parameters:
- `path` (required): Path to .qgs or .qgz file

### `add_vector_layer`
Add vector data layer to the project.

Parameters:
- `path` (required): Path to vector file
- `name` (optional): Custom layer name
- `style` (optional): "default" | "categorized" | "graduated" | "rule-based"
- `crs` (optional): Override CRS

### `add_raster_layer`
Add raster data layer to the project.

Parameters:
- `path` (required): Path to raster file
- `name` (optional): Custom layer name
- `colormap` (optional): "grayscale" | "terrain" | "rainbow" | "viridis"
- `min_max` (optional): [min, max] stretch values

### `run_processing`
Execute QGIS processing algorithm.

Parameters:
- `algorithm` (required): Algorithm ID (e.g., "native:buffer")
- `parameters` (required): Algorithm-specific parameters object
- `feedback` (optional): Enable progress feedback

### `execute_python`
Run arbitrary Python code in QGIS context.

Parameters:
- `script` (required): Python code to execute
- `return_data` (optional): Capture and return output

### `export_map`
Export current map view to image.

Parameters:
- `output_path` (required): Path to save image
- `resolution` (optional): DPI (default: 300)
- `width` (optional): Width in pixels (default: 1920)
- `height` (optional): Height in pixels (default: 1080)
- `format` (optional): "png" | "jpg" | "pdf" | "svg"

### `get_layer_info`
Get information about a layer.

Parameters:
- `layer_name` (required): Name of the layer
- `include_features` (optional): Include feature count and extent

---

## Common Processing Algorithms

### Geometry Operations
- `native:buffer` - Create buffer around features
- `native:centroid` - Calculate feature centroids
- `native:convexhull` - Create convex hull
- `native:envelope` - Create bounding boxes

### Overlay Analysis
- `native:intersection` - Intersect two layers
- `native:union` - Union two layers
- `native:difference` - Find difference between layers
- `native:clip` - Clip layer by polygon

### Vector Selection
- `native:extractbyattribute` - Extract features by attribute
- `native:extractbylocation` - Extract by spatial relationship
- `native:selectbylocation` - Select features spatially

### Raster Analysis
- `native:hillshade` - Create hillshade from DEM
- `native:slope` - Calculate slope
- `native:aspect` - Calculate aspect
- `native:rasterize` - Convert vector to raster

### Data Management
- `native:reprojectlayer` - Reproject layer to different CRS
- `native:dissolve` - Dissolve features by attribute
- `native:mergevectorlayers` - Merge multiple layers

---

## Examples

### Example 1: Urban Planning Analysis

```
Use QGIS skill to:
1. Load parcels shapefile from /data/parcels.shp
2. Load zoning layer from /data/zoning.geojson
3. Intersect parcels with zoning to find parcels in residential zones
4. Create 100m buffer around parks layer
5. Select parcels within park buffers
6. Export results to /output/parcels_near_parks.gpkg
7. Generate map with legend showing zoning categories
```

### Example 2: Environmental Analysis

```
Perform watershed analysis:
1. Load DEM raster from /data/elevation.tif
2. Calculate slope from DEM
3. Create hillshade for visualization
4. Load stream network from /data/streams.shp
5. Buffer streams by 30m
6. Clip vegetation layer to stream buffers
7. Calculate vegetation statistics within buffer
8. Export analysis map to /output/riparian_analysis.pdf
```

### Example 3: Batch Processing

```
Process multiple shapefiles:
1. List all .shp files in /data/counties/
2. For each county file:
   - Load layer
   - Reproject to UTM Zone 12N (EPSG:32612)
   - Dissolve by county name
   - Calculate area in square kilometers
   - Save to /output/counties_dissolved/
3. Merge all processed counties into single layer
4. Export combined layer to GeoPackage
```

---

## Troubleshooting

### Issue: QGIS Not Responding
**Symptoms**: Connection timeout or "QGIS not running" error
**Cause**: QGIS not running or socket server plugin not active
**Solution**:
```bash
# 1. Start QGIS Desktop
qgis &

# 2. Enable socket server plugin
# QGIS → Plugins → Manage and Install Plugins → Search "Server" → Enable

# 3. Verify socket is listening
netstat -an | grep 2801
```

### Issue: Layer Won't Load
**Symptoms**: "Failed to add layer" error
**Cause**: Invalid file path, unsupported format, or CRS mismatch
**Solution**:
```bash
# Check file exists and is readable
ls -l /data/layer.shp

# Verify format with ogr info
ogrinfo -summary /data/layer.shp

# Check CRS
ogrinfo /data/layer.shp | grep PROJCS
```

### Issue: Processing Algorithm Fails
**Symptoms**: Algorithm execution error
**Solution**:
```python
# List available algorithms
execute_python({
  "script": """
import processing
for alg in processing.algorithmHelp():
    print(alg)
""",
  "return_data": true
})

# Get algorithm help
execute_python({
  "script": "processing.algorithmHelp('native:buffer')",
  "return_data": true
})
```

### Issue: Invalid CRS
**Symptoms**: "Unknown CRS" or projection errors
**Solution**:
```bash
# Common CRS codes:
# EPSG:4326 - WGS84 (lat/lon)
# EPSG:3857 - Web Mercator
# EPSG:32612 - UTM Zone 12N
# EPSG:2227 - California State Plane

# Reproject layer
run_processing({
  "algorithm": "native:reprojectlayer",
  "parameters": {
    "INPUT": "MyLayer",
    "TARGET_CRS": "EPSG:4326",
    "OUTPUT": "/output/reprojected.shp"
  }
})
```

---

## Integration with Other Skills

Works well with:
- `filesystem` skill for managing geodata files
- `imagemagick` skill for post-processing exported maps
- `blender` skill for 3D terrain visualization
- `python` for custom geoprocessing scripts
- `postgresql` for PostGIS database operations

---

## Advanced Usage

### Custom Python Scripts

Execute complex QGIS operations:

```python
execute_python({
  "script": """
from qgis.core import QgsProject, QgsVectorLayer, QgsField
from PyQt5.QtCore import QVariant

# Get layer
layer = QgsProject.instance().mapLayersByName('Cities')[0]

# Add new field
layer.dataProvider().addAttributes([QgsField('density', QVariant.Double)])
layer.updateFields()

# Calculate population density
with edit(layer):
    for feature in layer.getFeatures():
        area = feature.geometry().area() / 1000000  # km²
        pop = feature['population']
        density = pop / area if area > 0 else 0
        feature['density'] = density
        layer.updateFeature(feature)

print(f'Updated {layer.featureCount()} features')
"""
})
```

### Spatial Queries

```python
# Find features within distance
execute_python({
  "script": """
from qgis.core import QgsProject, QgsFeatureRequest, QgsDistanceArea

layer = QgsProject.instance().mapLayersByName('Points')[0]
distance_calc = QgsDistanceArea()

# Find features within 1000m of a point
reference_point = QgsPointXY(-122.4194, 37.7749)
nearby = []

for feature in layer.getFeatures():
    dist = distance_calc.measureLine(reference_point, feature.geometry().asPoint())
    if dist <= 1000:
        nearby.append(feature['name'])

print(f'Found {len(nearby)} features within 1km')
"""
})
```

---

## Performance Notes

- Layer loading: < 1 second for small datasets (< 10K features)
- Processing algorithms: varies by complexity and data size
- Large datasets (> 100K features): consider spatial indexing
- Raster operations: memory-intensive, monitor RAM usage
- Socket communication overhead: ~50-100ms per operation

## Resources

- [QGIS Documentation](https://docs.qgis.org/)
- [PyQGIS Developer Cookbook](https://docs.qgis.org/latest/en/docs/pyqgis_developer_cookbook/)
- [Processing Algorithm List](https://docs.qgis.org/latest/en/docs/user_manual/processing_algs/)
- [QGIS Tutorials](https://www.qgistutorials.com/)

---

**Created**: 2025-10-20
**QGIS Version**: 3.28+ (LTS) or 3.34+
**Port**: 2801 (socket communication)
**Protocol**: JSON-RPC over TCP socket
