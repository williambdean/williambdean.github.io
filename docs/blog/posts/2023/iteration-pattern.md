---
tags: 
    - Design Patterns
    - Python
comments: true
---
# Iteration Pattern

Make use of python iteration


```python 
for leaf in tree: 
    print(leaf)
```

## Optional 1: Define in object


First way to achieve this is while making the model. The `__iter__` method is defined on the class and will be called when the object is iterated over.

For a binary tree, this would look like this: 

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Tree: 
    value: Any
    left: Tree = None
    right: Tree = None

    def __iter__(self): 
        yield self.value
        if self.left: 
            yield from self.left
        if self.right: 
            yield from self.right
```

Using this method, we can iterate over the tree and print the values. Example taken from [here](https://www.geeksforgeeks.org/difference-between-bfs-and-dfs/).

```text 
 Input:
        A
       / \
      B   C
     /   / \
    D   E   F
```

```python
tree = Tree("A", 
    Tree("B", 
        Tree("D"), 
    ), 
    Tree("C", 
        Tree("E"), Tree("F")
    )
)

for leaf in tree: 
    print(leaf)
```

```
1
3
6
2
5
4
```

## Optional 2: Making Custom Iterators

Using a generator function, we can define the iteration pattern outside of the class. This is useful if we want to define multiple iteration patterns for the same object. 

These look like functions that use the `yield` keyword instead of `return`. 


```python
def depth_first_search(tree): 
    stack = [tree]
    while stack: 
        leaf = stack.pop()
        if isinstance(leaf, Tree): 
            stack.extend([leaf.left, leaf.right])
        else: 
            yield leaf

def breadth_first_search(tree): 
    queue = [tree]
    while queue: 
        leaf = queue.pop(0)
        if isinstance(leaf, Tree): 
            queue.extend(leaf)
        else: 
            yield leaf
```

```python
for leaf in dfs(tree): 
    print(leaf)
```

```python
for leaf in bfs(tree): 
    print(leaf)
```

## Which to Use?

Most of the time we are handed different data structures and wouldn't want to override 