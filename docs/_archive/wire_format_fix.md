# Binary Wire Format Fix

## Problem
The server and client had a wire format mismatch:
- Server was using `WireNodeDataItem` with u32 ID (28 bytes total)
- Client expected u16 ID format (26 bytes total)
- This caused "BufferBoundsExceeded" errors and deserialization failures

## Solution
Fixed the wire format to be exactly 26 bytes as expected by the client:

### Wire Format Structure (26 bytes)
```
- id: u16 (2 bytes) - With node type flags in high bits
- position: Vec3Data (12 bytes = 3 * f32)
- velocity: Vec3Data (12 bytes = 3 * f32)
Total: 26 bytes
```

### Node Type Flags
For the u16 wire format, we use the high bits to indicate node type:
- Bit 15 (0x8000): Agent node
- Bit 14 (0x4000): Knowledge node
- Bits 0-13: Actual node ID (max 16,383 nodes)

### Key Changes

1. **Updated `WireNodeDataItem` structure** in `src/utils/binary_protocol.rs`:
   - Changed from u32 to u16 for ID field
   - Removed bytemuck Pod/Zeroable derives (not needed with manual serialization)
   - Added manual serialization/deserialization to ensure exact 26-byte format

2. **Added conversion functions**:
   - `to_wire_id(u32) -> u16`: Converts server u32 ID to wire u16 format, preserving flags
   - `from_wire_id(u16) -> u32`: Converts wire u16 ID back to server u32 format
   - `BinaryNodeData::to_wire_format()`: Converts server node data to wire format

3. **Updated encoding/decoding**:
   - Manual byte-level serialization instead of bytemuck to ensure exact format
   - Proper handling of little-endian byte order
   - Flag preservation during ID conversion

4. **Updated socket handler** in `src/handlers/socket_flow_handler.rs`:
   - Changed binary message size validation from 28 to 26 bytes
   - Updated comments to reflect correct wire format

## Testing
Added comprehensive tests to verify:
- Wire format is exactly 26 bytes
- Encoding/decoding roundtrip works correctly
- Node type flags are preserved in u16 format
- Maximum node ID (16,383) fits in 14 bits

## Backward Compatibility
The client continues to work unchanged as it was already expecting the 26-byte format.
The server now correctly sends data in the format the client expects.