#!/usr/bin/env python3
"""
TIF File Diagnostic Tool
Analyzes TIF file structure to debug loading issues
"""

import struct
import os
from pathlib import Path

def analyze_tif_file(file_path):
    """Analyze TIF file structure and report findings"""
    print(f"Analyzing TIF file: {file_path}")
    
    if not os.path.exists(file_path):
        print("ERROR: File does not exist")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    try:
        with open(file_path, 'rb') as f:
            # Read first 16 bytes for analysis
            header = f.read(16)
            
            if len(header) < 8:
                print("ERROR: File too small to be a valid TIFF")
                return
            
            # Check TIFF signature
            if header[:2] == b'II':
                endian = '<'
                endian_name = "Little Endian (Intel)"
            elif header[:2] == b'MM':
                endian = '>'
                endian_name = "Big Endian (Motorola)"
            else:
                print(f"ERROR: Invalid TIFF signature: {header[:2]}")
                print("This may not be a TIFF file or may be corrupted")
                return
            
            print(f"TIFF signature: Valid ({endian_name})")
            
            # Read magic number
            magic = struct.unpack(endian + 'H', header[2:4])[0]
            print(f"Magic number: {magic} ({'Valid' if magic == 42 else 'INVALID'})")
            
            if magic != 42:
                print("ERROR: Invalid TIFF magic number")
                return
            
            # Read first IFD offset
            ifd_offset = struct.unpack(endian + 'L', header[4:8])[0]
            print(f"First IFD offset: {ifd_offset}")
            
            if ifd_offset >= file_size:
                print("ERROR: IFD offset points beyond file end")
                return
            
            # Read IFD
            f.seek(ifd_offset)
            num_entries = struct.unpack(endian + 'H', f.read(2))[0]
            print(f"Number of IFD entries: {num_entries}")
            
            # Parse key tags
            tags_found = {}
            for i in range(num_entries):
                try:
                    tag, field_type, count, value_offset = struct.unpack(endian + 'HHLL', f.read(12))
                    tags_found[tag] = {
                        'type': field_type,
                        'count': count,
                        'value': value_offset
                    }
                except:
                    print(f"ERROR: Failed to read IFD entry {i}")
                    break
            
            # Report key image properties
            width = tags_found.get(256, {}).get('value', 'Unknown')
            height = tags_found.get(257, {}).get('value', 'Unknown')
            bits_per_sample = tags_found.get(258, {}).get('value', 'Unknown')
            compression = tags_found.get(259, {}).get('value', 'Unknown')
            photometric = tags_found.get(262, {}).get('value', 'Unknown')
            samples_per_pixel = tags_found.get(277, {}).get('value', 'Unknown')
            
            print(f"\nImage Properties:")
            print(f"  Width: {width}")
            print(f"  Height: {height}")
            print(f"  Bits per sample: {bits_per_sample}")
            print(f"  Compression: {compression} ({'Uncompressed' if compression == 1 else 'Compressed'})")
            print(f"  Photometric interpretation: {photometric}")
            print(f"  Samples per pixel: {samples_per_pixel}")
            
            # Check if we can handle this format
            can_handle = True
            issues = []
            
            if compression != 1:
                can_handle = False
                issues.append(f"Compressed format (compression={compression}) not supported in fallback mode")
            
            if bits_per_sample not in [8, 16]:
                can_handle = False
                issues.append(f"Unsupported bit depth: {bits_per_sample}")
            
            if 273 not in tags_found:  # StripOffsets
                can_handle = False
                issues.append("Missing StripOffsets tag")
            
            if 279 not in tags_found:  # StripByteCounts
                can_handle = False
                issues.append("Missing StripByteCounts tag")
            
            print(f"\nCompatibility Assessment:")
            if can_handle:
                print("✓ This TIFF file should be readable by our fallback reader")
            else:
                print("✗ This TIFF file has compatibility issues:")
                for issue in issues:
                    print(f"  - {issue}")
            
            # Show all tags for debugging
            print(f"\nAll TIFF tags found:")
            tag_names = {
                256: "ImageWidth",
                257: "ImageLength", 
                258: "BitsPerSample",
                259: "Compression",
                262: "PhotometricInterpretation",
                273: "StripOffsets",
                277: "SamplesPerPixel",
                278: "RowsPerStrip",
                279: "StripByteCounts",
                282: "XResolution",
                283: "YResolution",
                296: "ResolutionUnit"
            }
            
            for tag, data in tags_found.items():
                tag_name = tag_names.get(tag, f"Tag_{tag}")
                print(f"  {tag_name} ({tag}): type={data['type']}, count={data['count']}, value={data['value']}")
                
    except Exception as e:
        print(f"ERROR analyzing file: {e}")

def main():
    """Test with the uploaded file"""
    # Check for uploaded TIF files
    upload_dir = Path("attached_assets")
    tif_files = list(upload_dir.glob("*.TIF")) + list(upload_dir.glob("*.tif"))
    
    if tif_files:
        for tif_file in tif_files:
            print("="*60)
            analyze_tif_file(tif_file)
            print("="*60)
    else:
        print("No TIF files found in attached_assets directory")
        print("Available files:")
        if upload_dir.exists():
            for file in upload_dir.iterdir():
                print(f"  {file.name}")

if __name__ == "__main__":
    main()