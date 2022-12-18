---
tags: 
    - Python
    - Standard Library
---

# My Favorite Python builtin: pathlib

When working with python and data, file names quickly become a pain for many reasons. 

Firstly, files quickly add up. This could be from raw data, created processed data, config files, results including models and vizualizations. 

Using the strings can work, but this leads to working with many of the functions from the `os` and `os.path` modules.

```python 
import os 
from os.path import join

CURRENT_DIR: str = os.getcwd()

file_name: str = "my-data.csv"

data_file = join(CURRENT_DIR, file_name)

print(f"The file {data_file} exists: {os.path.isfile(data_file)}")
```

## Quick Start


## Using a data folder  

I often have a `DATA_DIR` constent for many of my projects setting up a folder `data` off the root of my project. 

```
my_module
data
```

```python title="my_module/utils.py"
from pathlib import Path

DATA_DIR = Path(__file__)
if not DATA_DIR.exists(): 
    DATA_DIR.mkdir()
```

I often extend this to include other folders I will likely have based on a project. This might be a folder `data/raw`, `data/results`, or even a `configs` dir.

## Working with S3 locations

I learned about this project while working with S3 paths on AWS. I haven't personally used but I think it looks promising and would provide a similar experience as the `pathlib` module.

https://github.com/liormizr/s3path