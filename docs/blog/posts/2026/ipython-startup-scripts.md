---
description: Tired of re-importing the same modules every time you open IPython? Here's how I use project-level startup scripts to jump straight into analysis.
tags:
    - Python
    - Development
    - Config Files
comments: true
---

# Project-Level IPython Startup Scripts

Every time I open IPython in a project directory, I want to hit the ground running. Not type imports. Not remember which random seed I used last time. Just work.

That's where project-level startup scripts come in.

## How It Works

IPython startup files live in `~/.ipython/profile_default/startup/`. Any `.py` or `.ipy` files in this directory are executed when IPython starts, in alphabetical order, so you can control execution order with prefixes like `00-`, `01-`, etc.

I symlink my dotfiles to keep them in sync:

```bash
ln -s ~/dotfiles/ipython ~/.ipython/profile_default/startup
```

This way, my startup scripts live in my dotfiles repo and travel with me across machines. My setup includes:

- `00-basic-imports.py` - global imports I want everywhere
- `01-custom-startup.py` - the project-level import trick

```python title="01-custom-startup.py"
try:
    from startup import *  # noqa: F401
except ImportError:
    pass
```

It tries to import from a local `startup.py` in the current directory, and silently fails if one doesn't exist. No errors, no fuss.

## The Magic: It Differs by Project

Here's where it gets useful. Each project can have its own `startup.py` with exactly what that project needs.

For a data analysis project, you might have:

```python title="startup.py (data project)"
"""Startup script to initialize data and random number generator."""

import myproject.data as md
import numpy as np

df = md.load_main_dataset()

seed = sum(map(ord, "analysis-2026"))
rng = np.random.default_rng(seed)
```

When I'm in that directory, IPython loads the dataset and sets up a reproducible RNG. I'm ready to analyze immediately.

For a different project, I might have:

```python title="startup.py (ml project)"
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
```

Or maybe a project that needs API clients, database connections, or ML experiment trackers.

## Why This Beats Global Startup Scripts

Global startup scripts are great for always-available imports like `pandas` or `numpy`. But project-level scripts give you something they can't: **context-specific setup**.

The data project `startup.py` knows about your dataset and uses a seed based on the project name. A generic data science starter can't provide that.

And because the import is wrapped in a try/except, there's no error when you're in a directory without a `startup.py`. It just falls back to your global imports.

!!! tip

    The numbered prefix controls execution order. Files run in alphabetical order, so use `00-` for global imports, `10-` for project-specific imports, etc.

## Further Inspiration

If you want to see how someone else sets theirs up, check out [Cam Davidson-Pilon's startup files](https://github.com/CamDavidsonPilon/StartupFiles/tree/master/startup). He uses a similar numbered prefix system with files like `00-imports.py`, `01-plotting.py`, `05-data_analysis.py`, and more. Great ideas if you're building out your own setup.

What imports do you find yourself typing at the start of every IPython session? Drop them in a `startup.py` and never type them again.
