---
description: Discover how the `--pdb` flag in pytest can be a game-changer for debugging Python tests, allowing you to drop into a debugger when a test fails.
tags: 
    - Python
    - Testing
comments: true
---

# Debugging with pytest

The `--pdb` flag in `pytest` is game changing for debugging tests. It allows you to drop into a debugger when a test fails. 

I've set up a command `DRunTests` to kick this off for the test that my cursor is on. 

If you would like to check around at a given point, use `assert 0` as an alternative. Here it is in action:

![type:video](../videos/pytest-debug.mp4)

There are a few great commands to remember when using the debugger:

- `p` to print variables
- `pp` to pretty print variables
- `q` to quit the debugger
- `c` to continue running the test

!!! tip

    When in doubt, use the `h(elp) expression` to get a list of all available commands.

Does the pytest --pdb flag come in handy for you? Let me know in the comments below! 
