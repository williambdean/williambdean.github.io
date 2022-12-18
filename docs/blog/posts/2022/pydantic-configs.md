---
tags: 
    - Python
    - Config Files
---

# Using PyDantic for Configs

I discovered the [pydantic library](https://docs.pydantic.dev/) when I first started using Typer and FastAPI and quickly found the library very useful for other reasons.

One usecase I've found very helpful is when making config files for python scripts. 

There is clear benefit to using configs when writing python code. i.e. you can change variables without having to edit the python file itself. But by also using pydantic you get the additional benefits provided from the library. 

## Clearly Define the Structure of the Config

When working with configs, I often find it confusing to what all the possible supported settings are. However, if you define a pydantic basemodel, you see clear instructions for what you are working with. 

For instance, a project which 


```python
from pydantic import BaseModel

from pathlib import Path
from typing import Dict


class Config(BaseModel):
    # Input  
    input_location: Path
    # Result location and file name
    results_dir: Path
    results_file_name: str
    plotting_kwargs: Dict

```

```yaml title="config.yaml"
input_location: ./data/input_data.csv
results_dir: .
results_file_name: my_first_run.png
plotting_kwargs: 
    alpha: 0.5
```

### Adding Hierarchy

If the config starts to get too long, then I've found splitting up into different sections to be very helpful. This can be sections for inputs, output, related setting, etc. 

```yaml 
input: 
    file: some-input-file.csv
output: 
    base_dir: some-directory-to-save
    result_name: some-file-name.png

```

Each one of the sections would be its own class in the defined pydantic model. 

```python 
class InputSetting(BaseModel): 
    file: Path 


class OutputSetting(BaseModel): 
    base_dir: Path
    results_name: str 


class Config(BaseModel): 
    """Class version of the full config file"""
    input: InputSettings
    output: OutputSettings

```

### Limiting Options

If you want to limit options, an `enum.Enum` type can be used to enforce only a set number of choices.

This provides some checking at config parsing time which can give you some quick feedback.

```python 
from enum import Enum 


class Difficulties(Enum): 
    EASY: str = "easy"
    MEDIUM: str = "medium"
    HARD: str = "hard"


class Config(BaseModel): 
    difficulty: Difficulties

```

Also just take advantage of all the added validation that pydantic provides. 

If you are used to using dataclasses too, the dataclasses submodule can be very helpful in order to add some additional checks on the configs at runtime.

```python 
from pydantic.dataclasses import dataclass


@dataclass
class ResultSettings: 
    results_dir: Path
    file_name: str 
    override: bool = False

    def __post_init__(self) -> None: 
        if not self.results_dir.exists(): 
            self.results_dir.mkdir()
        
        save_location = self.results_dir / self.file_name
        if save_location.exists() and not override: 
            msg = f"The results already exists. Not running {save_location}" 
            raise ValueError(msg)
```

## Class Implementation

I often add a lightweight class implementation when working with YAML configs. The goal here is to add an additional method to the pydantic `BaseModel` in order to easily load different config files.


``` py title="yaml_base_model.py"
from pydantic import BaseModel
import yaml

from pathlib import Path


class YamlBaseModel(BaseModel)
    @classmethod
    def from_yaml(cls, file: str | Path) -> YamlBaseModel: 
        file = Path(file)

        with open(file, "r") as f: 
            data = yaml.safe_load(f)

            return cls.parse_obj(data)

```

This allows for easy construction of a con

Find the gist of this [here](https://gist.github.com/wd60622/d8cc702f48e51d9a1e687ca1b6b66212) with an additional example.

## Conclusion

Overall, I've found defining configs with pydantic in mind very useful