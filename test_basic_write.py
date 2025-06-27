#!/usr/bin/env python3

# Test 1: Basic file writing
print("Testing basic file writing...")
try:
    with open('basic_test.txt', 'w', encoding='utf-8') as f:
        f.write("Hello, this is a test file\n")
        f.write("Second line\n")
        f.write("Third line\n")
    print("✓ Basic file written successfully")
except Exception as e:
    print(f"✗ Basic file writing failed: {e}")

# Test 2: CSV writing with basic approach
print("\nTesting basic CSV writing...")
try:
    with open('basic_test.csv', 'w', encoding='utf-8') as f:
        f.write("image_file,x0,y0,x1,y1\n")
        f.write("151.bmp,0.42678347,0.4397496,0.44055068,0.73552424\n")
        f.write("152.bmp,0.4192741,0.39749607,0.39299124,0.68701094\n")
    print("✓ Basic CSV written successfully")
except Exception as e:
    print(f"✗ Basic CSV writing failed: {e}")

# Test 3: Check file sizes
import os
print("\nChecking file sizes...")
if os.path.exists('basic_test.txt'):
    size = os.path.getsize('basic_test.txt')
    print(f"basic_test.txt size: {size} bytes")
if os.path.exists('basic_test.csv'):
    size = os.path.getsize('basic_test.csv')
    print(f"basic_test.csv size: {size} bytes") 