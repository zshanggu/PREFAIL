#!/usr/bin/env python3
import os
import sys
import re
import argparse

def extract_object_names_with_debug(directory):
    """
    Extract object names from NPY files with detailed debugging info.
    """
    processed_objects = set()
    all_npy_files = []
    matched_files = []
    unmatched_files = []
    total_files = 0
    total_dirs = 0
    
    # Regular expression to extract object name from filename
    pattern = r'episode\d+_(.*?)_train_\d+_\d+_raw\.npy'
    
    # Walk through the directory
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            total_dirs += 1
            total_files += len(files)
            
            print(f"Directory: {root}")
            print(f"  Total files in this dir: {len(files)}")
            npy_count_in_dir = sum(1 for f in files if f.endswith('.npy'))
            print(f"  .npy files in this dir: {npy_count_in_dir}")
            
            for file in files:
                if file.endswith('.npy'):
                    full_path = os.path.join(root, file)
                    all_npy_files.append(full_path)
                    
                    match = re.search(pattern, file)
                    if match:
                        object_name = match.group(1)
                        processed_objects.add(object_name)
                        matched_files.append(full_path)
                    else:
                        unmatched_files.append(full_path)
    else:
        print(f"Directory: {directory} not found")
        return set(), [], [], [], 0, 0
    
    return processed_objects, all_npy_files, matched_files, unmatched_files, total_files, total_dirs

def main():
    parser = argparse.ArgumentParser(description='Extract object names from NPY files with debugging')
    parser.add_argument('--npy_dir', required=True, help='Directory containing NPY files')
    parser.add_argument('--output', required=True, help='Output file to store object names')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Extract object names with debug info
    processed_objects, all_npy_files, matched_files, unmatched_files, total_files, total_dirs = extract_object_names_with_debug(args.npy_dir)
    
    # Print file counts
    print(f"\n=== FILE COUNTS ===")
    print(f"Total directories searched: {total_dirs}")
    print(f"Total files (all types): {total_files}")
    print(f"Total .npy files found: {len(all_npy_files)}")
    print(f"Files matching pattern: {len(matched_files)}")
    print(f"Files NOT matching pattern: {len(unmatched_files)}")
    print(f"Unique objects extracted: {len(processed_objects)}")
    
    if args.debug:
        print("\n=== DEBUG INFO ===")
        
        # Show directory structure
        print(f"\nSearching in directory: {args.npy_dir}")
        print("Subdirectories found:")
        for root, dirs, files in os.walk(args.npy_dir):
            if files:
                npy_count = sum(1 for f in files if f.endswith('.npy'))
                if npy_count > 0:
                    print(f"  {root}: {npy_count} .npy files")
        
        # Show unmatched files (first 10)
        if unmatched_files:
            print(f"\nFirst 10 files that DON'T match pattern:")
            for file in unmatched_files[:10]:
                print(f"  {os.path.basename(file)}")
        
        # Show some matched objects
        if processed_objects:
            print(f"\nFirst 10 object names extracted:")
            for obj in list(processed_objects)[:10]:
                print(f"  {obj}")
    
    # Write to output file
    with open(args.output, 'w') as f:
        for obj in sorted(processed_objects):
            f.write(f"{obj}\n")
    
    print(f"\nSaved {len(processed_objects)} unique objects to {args.output}")

if __name__ == "__main__":
    main()