#!/usr/bin/env python3
"""
Create a simple WAV file with a sine wave tone for testing
"""
import struct
import math

def create_wav_file(filename, duration_seconds=2, sample_rate=16000, frequency=440):
    """Create a simple WAV file with a sine wave"""
    
    # Calculate number of samples
    num_samples = int(sample_rate * duration_seconds)
    
    # Generate sine wave data
    samples = []
    for i in range(num_samples):
        t = float(i) / sample_rate
        value = int(32767 * math.sin(2 * math.pi * frequency * t))
        samples.append(value)
    
    # WAV file header
    with open(filename, 'wb') as wav_file:
        # RIFF header
        wav_file.write(b'RIFF')
        file_size = 36 + num_samples * 2  # header + data
        wav_file.write(struct.pack('<I', file_size))
        wav_file.write(b'WAVE')
        
        # fmt chunk
        wav_file.write(b'fmt ')
        wav_file.write(struct.pack('<I', 16))  # fmt chunk size
        wav_file.write(struct.pack('<H', 1))   # PCM format
        wav_file.write(struct.pack('<H', 1))   # 1 channel (mono)
        wav_file.write(struct.pack('<I', sample_rate))
        wav_file.write(struct.pack('<I', sample_rate * 2))  # byte rate
        wav_file.write(struct.pack('<H', 2))   # block align
        wav_file.write(struct.pack('<H', 16))  # bits per sample
        
        # data chunk
        wav_file.write(b'data')
        wav_file.write(struct.pack('<I', num_samples * 2))
        
        # Write samples
        for sample in samples:
            wav_file.write(struct.pack('<h', sample))
    
    print(f"Created {filename}: {duration_seconds}s, {sample_rate}Hz, {frequency}Hz tone")
    print(f"File size: {file_size + 8} bytes")

if __name__ == "__main__":
    create_wav_file("/tmp/test_tone.wav", duration_seconds=2, sample_rate=16000, frequency=440)
    print("\nFile created successfully!")