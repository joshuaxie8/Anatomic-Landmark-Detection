#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Test data
test_data = {
    'image_file': ['test1.bmp', 'test2.bmp'],
    'x0': [0.5, 0.6],
    'y0': [0.3, 0.4],
    'x1': [0.7, 0.8],
    'y1': [0.2, 0.3]
}

# Create DataFrame
df = pd.DataFrame(test_data)

# Save with UTF-8 encoding
df.to_csv('test_predictions.csv', index=False, encoding='utf-8')

print("Test CSV file created successfully!")
print("File contents:")
print(df)

# Try to read it back
df_read = pd.read_csv('test_predictions.csv', encoding='utf-8')
print("\nRead back successfully:")
print(df_read) 