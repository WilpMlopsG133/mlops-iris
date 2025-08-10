import os
import re

mlruns_path = 'mlruns'
old_path_prefix = 'file:///C:'
new_path_prefix = 'file:///app'

for root, dirs, files in os.walk(mlruns_path):
    for file in files:
        if file.endswith('.json') or file.endswith('.yaml'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()

            updated_content = re.sub(old_path_prefix, new_path_prefix, content)

            if updated_content != content:
                print(f"Fixing path in {filepath}")
                with open(filepath, 'w') as f:
                    f.write(updated_content)

print("Path fixing complete.")

