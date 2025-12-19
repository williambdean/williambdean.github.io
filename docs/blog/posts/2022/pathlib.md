---
description: A look at the pathlib module in the Python standard library and how it can be used to work with files and paths in a more intuitive and object-oriented way.
tags: 
    - Python
    - Standard Library
comments: true
---

# My Favorite Python builtin: pathlib

When working with python and data, file names quickly become a pain for many reasons. 

Firstly, files quickly add up. This could be from raw data, created processed data, config files, results including models and vizualizations. A lot more files are being worked with than initially thought, so not being organized can be quickly overwhelming.

Secondly — and this may be a personal problem for me — start adding lengthy, hard-coded values for file names just becomes painful. Sometimes I even find it difficult to want to start working scripts.

Of course, using strings can work, but this leads to working with many of the functions from the `os` and `os.path` modules. Who wants so many imports too? 

```python 
import os 
from os.path import join

CURRENT_DIR: str = os.getcwd()

file_name: str = "my-data.csv"

data_file = join(CURRENT_DIR, file_name)

print(f"The file {data_file} exists: {os.path.isfile(data_file)}")
```

Now, hand off your scripts to someone on a Window's machine. All your hard-coded paths aren't even in the right format now and I'm feeling total regrets for ever starting the project :smile:.

## Quick Start

Using a `pathlib.Path` instance instead of a string is meant to be intuitive. 

That is, many actions like creating file, checking stats, relative location, etc are just methods of a `Path` instance. Other things like file name, suffix, or parents are attributes of the instance. Forget all the imports, working with `Path` object takes advantage of OOP design.

When working in a python file, the `__file__` variable can be utilized in order to find out where the current location is. No need to hard code or think about relative paths.

```python title="current-file.py"
from pathlib import Path 


HERE = Path(__file__).parent

# New file next to "current-file.py"
new_file = HERE / "new-file.txt"

if not new_file.exists(): 
    new_file.touch()

new_file.write_text("Writing some text to the new file!")

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir()

# Some processing
...
override: bool
result_file = RESULTS_DIR / "results-file.csv"
if results_file.exists() and not override: 
    msg = "We don't want to override this file!"
    raise ValueError(msg)
```

I find the interface very intuitive and cool thing here is that this will work on the machine regardless of the operating system. 


## Using a data folder  

I often have a `DATA_DIR` constant for many of my projects which refers to a folder `data` off the root of my project. 

In the file system, that would look like this:

```
my_module/
    ...
data/
    ...
README.md
```

This is an easy setup and saves a lot of headaches in the future.

```python title="my_module/utils.py"
from pathlib import Path

DATA_DIR = Path(__file__)
if not DATA_DIR.exists(): 
    DATA_DIR.mkdir()
```

As long as I am working with my python module, I don't have to worry about much more than the file names I want.

```python
from my_module import utils

file: Path = utils.DATA_DIR / "raw" / "my-data.csv"
```

### Additional folders

I often extend this to include other folders I will likely have based on a project. This might be a folder `data/raw`, `data/results`, or even a `configs` dir.

This depends on the project but all of this is with the goal of being as organized as possible from the start. 

I recently watched this related video on naming files and found it useful. 
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/ES1LTlnpLMk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Working with S3 locations

I learned about [this python package](https://github.com/liormizr/s3path) while working with S3 paths on AWS. I haven't personally used but I think it looks promising and would provide a similar enjoyable experience as the `pathlib` module.

## Conclusion

Taming all the files you are working with is never an easy battle. However, I find using `pathlib` makes the process just a little more enjoyable.