#!/usr/bin/env python3
import csv
import pandas as pd
import numpy as np

def test_csv_writing():
    print("=== Testing CSV Writing ===")
    
    # Create dummy data similar to what the prediction would produce
    dummy_data = {
        'image_file': ['151.bmp', '152.bmp', '153.bmp'],
        'x0': [0.42678347, 0.4192741, 0.4123456],
        'y0': [0.4397496, 0.39749607, 0.4567890],
        'x1': [0.44055068, 0.39299124, 0.4234567],
        'y1': [0.73552424, 0.68701094, 0.7123456],
        'x2': [0.54443055, 0.5006258, 0.5345678],
        'y2': [0.6682316, 0.63380283, 0.6456789]
    }
    
    # Test 1: Using pandas
    print("\n1. Testing pandas to_csv...")
    try:
        df = pd.DataFrame(dummy_data)
        df.to_csv('test_pandas.csv', index=False, encoding='utf-8')
        print("✓ Pandas CSV created successfully")
        
        # Try to read it back
        df_read = pd.read_csv('test_pandas.csv', encoding='utf-8')
        print(f"✓ Pandas CSV read back successfully, shape: {df_read.shape}")
        
    except Exception as e:
        print(f"✗ Pandas CSV failed: {e}")
    
    # Test 2: Using built-in csv module
    print("\n2. Testing built-in csv module...")
    try:
        with open('test_csv_module.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(dummy_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data rows
            for i in range(len(dummy_data['image_file'])):
                row = {key: dummy_data[key][i] for key in fieldnames}
                writer.writerow(row)
        
        print("✓ CSV module file created successfully")
        
        # Try to read it back
        with open('test_csv_module.csv', 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"✓ CSV module file read back, first line: {first_line}")
            
    except Exception as e:
        print(f"✗ CSV module failed: {e}")
    
    # Test 3: Check file contents
    print("\n3. Checking file contents...")
    try:
        with open('test_pandas.csv', 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"✓ Pandas file content length: {len(content)}")
            print(f"First 100 chars: {content[:100]}")
            
    except Exception as e:
        print(f"✗ Could not read pandas file: {e}")
    
    try:
        with open('test_csv_module.csv', 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"✓ CSV module file content length: {len(content)}")
            print(f"First 100 chars: {content[:100]}")
            
    except Exception as e:
        print(f"✗ Could not read CSV module file: {e}")

if __name__ == "__main__":
    test_csv_writing() 