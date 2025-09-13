#!/usr/bin/env python3
import json

metadata_path = "/workspace/ext/data/metadata/metadata.json"

try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        print(f"Total entries in metadata.json: {len(metadata)}")
        
        # Show first 3 entries
        for i, (key, value) in enumerate(metadata.items()):
            if i >= 3:
                break
            print(f"\nEntry {i+1}:")
            print(f"  Key: {key}")
            print(f"  fileName: {value.get('fileName', 'N/A')}")
            print(f"  fileSize: {value.get('fileSize', 'N/A')}")
            print(f"  nodeSize: {value.get('nodeSize', 'N/A')}")
            
except Exception as e:
    print(f"Error reading metadata: {e}")