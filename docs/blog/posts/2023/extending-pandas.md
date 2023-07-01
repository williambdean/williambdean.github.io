---
tags: 
    - Python
    - Pandas
    - Data Analysis
comments: true
---
# Extending Pandas

Here is a quick way to make your functionality with pandas objects just as common as using pandas itself and help promote readable code. 

Our goal here is to make a function that can be used on a pandas object via a self defined attribute. 

```python
import pandas as pd
import my_module

df = pd.DataFrame(...)
# Using user defined boot attribute with its get_samples method
df_bootstrap: pd.DataFrame = df.boot.get_samples(my_func, B=100)
```

## Implementation

In order to do this, we need to create a class that extends the pandas object. This is done by using the `pd.api.extensions.register_dataframe_accessor` decorator. The name we pass will be the name of the attribute we use to access the functionality.

Below creates a functions that will bootstrap a function on a DataFrame and will define the `boot` attribute on a DataFrame with the `BootAccessor` class. There the bootstrap function is defined as a method on the class.

```python 
import pandas as pd

def bootstrap(df: pd.DataFrame, b_func, B: int = 100) -> pd.DataFrame: 
    """Bootstrap a function on a DataFrame. 

    Adds sample index to the result.
    
    Args:
        df (pd.DataFrame): DataFrame to bootstrap
        b_func (Callable): Function to bootstrap
        B (int, optional): Number of bootstrap samples. Defaults to 100.
    
    Returns:
        pd.DataFrame: DataFrame of bootstrap samples

    """
    return pd.concat([
        df
        .sample(frac=1, replace=True)
        .pipe(b_func)
        .rename(i)
        .to_frame() 
        for i in range(B)
    ], axis=1).T

@pd.api.extensions.register_dataframe_accessor("boot")
class BootAccessor: 
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def get_samples(self, b_func, B: int = 100) -> pd.DataFrame: 
        """Bootstrap a function on a DataFrame
        
        Args:
            b_func (Callable): Function to bootstrap
            B (int, optional): Number of bootstrap samples. Defaults to 100.
        
        Returns:
            pd.DataFrame: DataFrame of bootstrap samples

        """
        return bootstrap(self._obj, b_func=b_func, B=B)
```

## Usage

After import of this module, you can use the `boot` accessor on any pandas object.

```python
import pandas as pd

df = pd.DataFrame(...)
try: 
    df.boot
except AttributeError: 
    pass

import my_module

def my_func(df: pd.DataFrame) -> pd.Series: 
    """Function to bootstrap
    
    Args:
        df (pd.DataFrame): DataFrame to bootstrap
    
    Returns:
        pd.Series: mean of the columns
    """
    return df.mean()

df.boot.get_samples(b_func=my_func, B=100)
```

Though is just a single method, this technique can be used to package up a lot of functionality.

## Adding Validation

The `BootAccessor` class can be extended to add validation to the DataFrame before the bootstrap is performed. This can be good for checking that the DataFrame has the correct columns or that the values are in the correct range -- or anything else for the use case.

```python   
@pd.api.extensions.register_dataframe_accessor("boot")
class BootAccessor: 
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validation(df: pd.DataFrame) -> bool: 
        """Validate DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame to validate
        
        Returns:
            bool: True if DataFrame is valid
        """
        return True
```

A simple addition to add checks to all of your functionality.

## Alternatives & Conclusion

Using the `pipe` method on pandas objects is great way to make some readable code, but it can quickly become a bit verbose with imports.

```python
from my_module import bootstrap, preprocess_func, postprocess_func, plot_timeseries

df = pd.DataFrame(...)
df_result = (
    df
    .pipe(preprocess_func, ...)
    .pipe(bootstrap, b_func=my_func, B=100)
    .pipe(postprocess_func, ...)
    .pipe(plot_timeseries, ...)
```

An alternative might look like this

```python 
import my_module

df_result = (
    df
    .transformations.preprocess(...)
    .boot.get_samples(b_func=b_func, B=100)
    .transformations.postprocess(...)
    .plotting.timeseries(...)
)

```

All in all, it's a quick change to add new functionality the widely used data type and maybe help the user experience.

## Resources

- [Pandas User Guide: Extending Pandas](https://pandas.pydata.org/docs/development/extending.html)

