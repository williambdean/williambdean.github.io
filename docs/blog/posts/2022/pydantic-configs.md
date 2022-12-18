---
tags: 
    - Python
    - Config Files
---

# Using PyDantic for Configs

!!! tip 
    It might be helpful to know a bit about the pydantic library and its functionality. Luckily, the docs are very good. They are linked below.

I discovered the [pydantic library](https://docs.pydantic.dev/) when I first started using [Typer](https://typer.tiangolo.com/) and [FastAPI](https://fastapi.tiangolo.com/) and quickly found the library very useful for other reasons.

One use case I've found very helpful is when making config files for python scripts. 

There is clear benefit to using configs when writing python code: Variables can be changed without having to edit the python file itself. But by also using pydantic you get the additional benefits provided from the library. 

## Clearly Define the Structure of the Config

When working with configs, I often find it confusing to what all the possible supported settings are. However, if you define a `Config` class from the `pydantic.BaseModel`, you see clear structure for what you are working with. 

For instance, a project which is working with some input, output, and some additional configuration settings might look like this.

```python
from pydantic import BaseModel

from pathlib import Path
from typing import Dict, Any


class Config(BaseModel):
    # Input  
    input_location: Path
    # Result location and file name
    results_dir: Path
    results_file_name: str
    # Some additional configurations
    plotting_kwargs: Dict[str, Any] = {}

```

This data can come from many sources but a YAML configuration file for this structure might look like this:

```yaml title="config.yaml"
input_location: ./data/input_data.csv
results_dir: ./results/
results_file_name: my_first_run.png
plotting_kwargs: 
    alpha: 0.5
```

By looking at the class structure, it is clear that some input and output information is required and additional plotting information is optional. Not only that, the user gets an understanding of what format the data should be in.

### Adding Hierarchy

If the config starts to get too large, I've found splitting up into different sections to be very helpful. This can be sections for inputs, output, related setting, etc. A more organized YAML could look like this:

```yaml 
input: 
    file: some-input-file.csv
output: 
    base_dir: some-directory-to-save
    result_name: some-file-name.png

```

In order to support this structure, each one of the sections would be its own class in the defined pydantic model. 

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

This allows for like items to be broken up in a logical way and support complicated configuration options.

### Additional Functionality

Since all of these configs items are python classes, additional functionality can be added to them with methods and attributes. This add to the cohesion of the code by putting similiar methods together.

For instance, if there is some type of connection settings, and adding additional functionality to load data might be helpful to group together. 


```python 
import pandas as pd


class DataBaseSettings(BaseModel): 
    schema: str 
    table: str 

    def is_connected(self) -> bool: 
        """Determine if connection exists."""

    def read_table(self) -> pd.DataFrame: 
        """Return the table from the database."""

    
class Config(BaseModel): 
    database: DataBaseSettings
```

### Generalization and Extensions

Because of the powerful structure parsing of pydantics, we can extend our configs very easily by providing abstractions to our data models.

For instance, if multiple different data sources want to be supported, then that can be reflected in our configuration.

```python 
class CSVSettings(BaseModel): 
    """Data source that is csv format."""
    location: Path

    def read_table(self) -> pd.DataFrame: 
        """Read from csv file."""


class Config(BaseModel): 
    """Generalized configuration."""
    source: CSVSettings | DataBaseSettings


```

Because pydantic will be able to understand these structural differences, we are able to change our config file accordingly. 

```yaml 
---
source: 
    location: data/some-local-data.csv
---
source: 
    schema: my-schema
    table: my-table

```

When the config is parsed into an instance, the common interface can be leveraged in the code while also providing flexibility in the settings.

### Reusability

If there are multiple configuration files required for a project, there will often be overlapping configuration elements. By structuring the code in the [hierarchical manner](#adding-hierarchy), different classes can be reused in order to simplify our interface.

```python 
class RunConfig(BaseModel): 
    """Running and saving off a model."""
    input_settings: InputSettings
    model_settings: ModelSettings
    results: ResultsLocation


class InterpretationConfig(BaseModel): 
    """Loading and interpreting model."""
    results: ResultsLocation
```

The `ResultsLocation` class might be useful in multiple configuration files here because it is used to both save and load data.

### Limiting Options

If you want to limit options a variable can take, an `enum.Enum` type can be used to enforce only a set number of choices.

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

Or like the [other example above](#generalization-and-extensions), a union of types can be expressed.

### Additional Validation

Pydantic provides a bunch of [additional data validation](https://docs.pydantic.dev/usage/validators/) which can provide some runtime checks to your configuration.

If you are used to using dataclasses too, the dataclasses submodule can be very helpful in order to add some additional checks on the configs at runtime as well.

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

Then when defining a config file, this will be the class inherited from. Making it clear which define the structure of config files and which are just parts of a larger configuration.

```python
class ModelSettings(BaseModel): 
    """Won't be a config file but will be part of some larger configuration."""
    folds: int 
    method: str
    ...


class RunConfig(YamlBaseModel):
    """Some YAML config file will have this structure."""
    input: InputSettings
    model_settings: ModelSettings
```

This allows for easy construction of a config object and can be used accordingly.

```python title="run_script.py"

if __name__ == "__main__": 
    config = RunConfig.from_yaml("./configs/run-config.yaml")

    data = config.input.load_data()

```

Find the gist of this [here](https://gist.github.com/wd60622/d8cc702f48e51d9a1e687ca1b6b66212) with an additional example.

Prefer [TOML Configs](https://toml.io/en/)? Can imagine similar support for TOML configs (especially with latest support in python 3.11). Same goes with some additional formats too.

## Conclusion

Overall, I've found defining configs with pydantic in mind very useful. It can be super quick to do, provide a lot more structure and understanding to the config settings, and leverage the powerful parsing validation from the library. Give it a try!