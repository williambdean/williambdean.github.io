---
tags: 
    - Design Patterns
    - Python
    - Data Analysis
comments: true
---
# Iteration Pattern

Make use of python iteration for more generalized code and functionality

```python 
for value in iterable: 
    process(value)
```

## Allowing iteration in python

Objects can be used in `for` loop as long as the object is iterable or has a way to iterate over it.

This functionality can be defined with with the `__iter__` and `__next__` method or a generator. 

### 1. Generator Function

A generator is a function that returns an iterator. This is done by using the `yield` keyword instead of `return`.

```python title="Example of generator"
def generator(): 
    yield 1
    yield 2
    yield 3

def same_generator(): 
    for i in range(1, 4): 
        yield i

for value in generator():
    print(value)
```

```
1
2
3
```

!!! Note 
    Iterators are pretty common in python. i.e. `range` is an iterator.

!!! Tip "Alternatives"

    The `yield from` keyword can be used as an alternative to yield all values from another iterator.

    ```python title="Example of yield from"

    def generator(): 
        yield from range(1, 4)
    ```

    Another way to created a generator is with a generator expression.

    ```python title="Example of generator expression"
    generator = (i for i in range(1, 4))

    for value in generator: 
        print(value)
    ```

### 2. Define `__iter__` and `__next__` methods

The `__iter__` method returns an iterator object and the `__next__` method returns the next value in the iterator. The `StopIteration` exception is raised when there are no more values to return.

!!! Note 
    The class variables can be used to keep track of the current value and the max value.

```python title="Example of __iter__ and __next__ method"
class Iterator: 
    def __init__(self, max_value): 
        self.max_value = max_value
        self.current_value = 0

    def __iter__(self): 
        return self

    def __next__(self): 
        if self.current_value >= self.max_value: 
            raise StopIteration

        self.current_value += 1
        return self.current_value

for value in Iterator(3):
    print(value)
```

```
1
2
3
```

!!! Note 
    The return value of `__iter__` must be an iterator. This can be done by returning `self` or by defining a separate iterator class, generator function, or generator expression.

## Which to Use?

Most of the time we are handed different data structures and wouldn't want to override the `__iter__` method. In this case, we can use generator functions to define the iteration method. 

```python title="Sample Data Structure Handed to Us"
from dataclasses import dataclass

@dataclass
class Matrix: 
    data: list[list[int]]

    def __post_init__(self) -> None: 
        assert self.nrows > 0, "Matrix must have at least one row"
        assert self.ncols > 0, "Matrix must have at least one column"
        # Matrix must be rectangular
        assert all(len(row) == self.ncols for row in self.data), "Matrix must be rectangular"

    @property 
    def nrows(self) -> int: 
        return len(self.data)
        
    @property 
    def ncols(self) -> int: 
        return len(self.data[0])
```

Defining iteration outside of the object allows us to define different iteration methods. Any way we want to iterate over the matrix can be defined as a generator.

Below are three different methods for the `Matrix` class. 

1. Row First
2. Column First
3. Diagonal

```python title="Define Iteration Methods"

def row_first_iteration(matrix: Matrix): 
    for row in matrix.data: 
        for value in row: 
            yield value 

def column_first_iteration(matrix: Matrix): 
    for col in range(matrix.ncols): 
        for row in range(matrix.nrows): 
            yield matrix.data[row][col]

def diagonal_iteration(matrix: Matrix): 
    """Iterate through the matrix diagonally. 
    
    Starts with the top left corner first and goes diagonally up.
    
    """
    nrows = matrix.nrows
    ncols = matrix.ncols

    ndiags = ncols + nrows - 1

    for diag in range(n_diags):
        for col in range(diag + 1): 
            row = diag - col
            if row < nrows and col < ncols: 
                yield matrix.data[row][col]
```

Now when we want to iterate over the matrix, we can choose which iteration method to use depending on the use case.

```python title="Example Usage"
data = [
    [1, 2],
    [3, 4], 
    [5, 6], 
    [7, 8]
]
matrix = Matrix(data)

row_first_iter = row_first_iteration(matrix)
column_first_iter = column_first_iteration(matrix)
diag_iter = diagonal_iteration(matrix)

import pandas as pd 

df = pd.DataFrame({
    "Row First": list(row_first_iter),
    "Column First": list(column_first_iter),
    "Diagonal": list(diag_iter)
})
df.index.name = "Iteration"
```

```text
           Row First  Column First  Diagonal
Iteration                                   
0                  1             1         1
1                  2             3         3
2                  3             5         2
3                  4             7         5
4                  5             2         4
5                  6             4         7
6                  7             6         6
7                  8             8         8
```

The `for` loop will be the same regardless of how we want to process the values:

```python title="Various ways to process values"
process = print
iterable = row_first_iter 

for value in iterable:
    process(value)

# Alternative iteration and processing
def log_value_somewhere(value): 
    print(f"Logging value {value}")

process = log_value_somewhere
iterable = diag_iter

for value in iterable: 
    process(value)
```

This can be useful when we want to process the values in different ways, but also from different data structures. 

```python title="Example of different data structures"
import numpy as np

matrix_np = np.array(data)

iterable = iter(matrix_np.flatten())

for value in iterable: 
    process(value)
```


## Summary

Classes are often handed to next user so it useful to define iteration methods outside of the class. Not only that, but what is done with the value is separated from the iteration method as well. That is, separation of: 

1. Data 
2. Iteration of data
3. Processing of data

This provides flexibility while keeping the code in a consistent format.

```python
for value in iterable: 
    process(value)
```

## References

- [Python Generators](https://wiki.python.org/moin/Generators)
- [`__iter__` and `__next__`](https://www.geeksforgeeks.org/python-__iter__-__next__-converting-object-iterator/)