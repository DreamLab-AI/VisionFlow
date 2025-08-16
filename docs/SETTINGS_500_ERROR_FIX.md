# Settings 500 Error Fix

## Problem
The settings API was returning 500 Internal Server Error with the message:
```
Failed to merge settings: Failed to deserialize merged settings: invalid type: null, expected a string
```

## Root Cause
The `convert_empty_strings_to_null` function was incorrectly converting empty strings to null for fields that are required (non-Option) String fields in the Rust structs. When deserializing, Rust expected a string but got null, causing the error.

## Solution
Updated the `convert_empty_strings_to_null` function in `/workspace/ext/src/config/mod.rs` to:
1. Maintain a list of required string fields that should NOT be converted to null
2. Only convert empty strings to null for optional fields
3. Keep empty strings as-is for required fields

## Required String Fields
The following fields must always have a string value (even if empty):
- Colour fields: `base_colour`, `colour`, `background_colour`, `text_colour`, etc.
- Quality/mode fields: `quality`, `mode`, `context`, `billboard_mode`
- Network fields: `bind_address`, `domain`, `min_tls_version`, `tunnel_id`
- Auth fields: `provider`
- XR fields: Various colour and configuration strings

## Testing
After this fix, the settings API should:
1. Accept updates with empty strings for required fields
2. Convert empty strings to null only for optional fields
3. Successfully merge and save settings without 500 errors