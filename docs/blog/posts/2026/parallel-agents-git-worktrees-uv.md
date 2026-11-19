---
description: "How migrating from conda to uv for speed and simplicity unlocked an unexpected benefit: clean git worktree workflows for parallel AI coding agents."
tags:
    - Development
    - GitHub Actions
    - uv
    - Git
comments: true
---

# Parallel AI Agents with Git Worktrees and uv

!!! tip "TL;DR"
    We migrated [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) from conda to uv for faster dependency resolution and simpler CI: pymc 6.0 ships on PyPI, so conda-forge is no longer needed. The unexpected benefit: uv's `--directory` flag pairs naturally with git worktrees. Each parallel AI coding agent gets its own branch, checkout, and committed `uv.lock`, invoked from anywhere with no `cd` and no shared named environments.

Recently we migrated [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) from
conda to uv. The reasons were straightforward: faster dependency resolution and
simpler CI. The key enabler: **[pymc 6.0.0](https://github.com/pymc-devs/pymc/releases/tag/v6.0.0) now ships on PyPI**, eliminating the
need for conda-forge entirely.

What we didn't expect was how it would unlock a workflow that's
becoming essential for AI-assisted development.

## The Migration

The migration (commit
[`681adae7`](https://github.com/pymc-labs/pymc-marketing/commit/681adae7), PR
[#2611](https://github.com/pymc-labs/pymc-marketing/pull/2611)) deleted
`environment.yml`, added `uv.lock`, and swapped micromamba for
`astral-sh/setup-uv` in CI. Makefile targets went from `pip install .[lint]` to
`uv sync --extra lint`. Contributing docs went from `conda env create -f
environment.yml` to `uv sync`.

The result was immediate: `uv sync` is noticeably faster than `conda env
create`, the lockfile is deterministic and committed, and CI workflows are
shorter and more reliable.

## The Unexpected Bonus

The [`--directory` flag](https://docs.astral.sh/uv/reference/cli/#uv-auth-login--directory) in uv
lets you point any command at an arbitrary project directory:

```bash
uv sync --directory /path/to/project
uv run --directory /path/to/project pytest
```

It works consistently across every subcommand: `uv sync`, `uv run`, `uv add`,
and `uv lock`. You can use it from anywhere with no `cd` required.

You could also `cd <worktree> && uv run pytest`. The `--directory` flag just
saves a step. The real win is that each worktree branch carries its own
`uv.lock`. Agents working on different branches can add dependencies, bump
versions, or change resolution strategies without conflicting. With the old
conda setup using micromamba, environments were named and shared, far less
practical for this kind of parallelism.

Point `--directory` at any directory with a `uv.lock` and you have an
isolated, reproducible environment. Git worktrees are a natural way to create
those directories: each branch gets its own checkout with its own lockfile.

## Git Worktrees and the Missing Piece

Git worktrees give you isolated checkouts for parallel development. Each
worktree lives in its own directory on its own branch. They're the standard
answer to running multiple AI coding agents on the same repository. Each agent
gets a worktree: no interference, true parallelism.

My setup follows a simple convention: worktrees live in a `worktrees/`
directory at the repo root, named after the task:

```
worktrees/fix-builder-pr
worktrees/add-analytics-endpoint
worktrees/upgrade-deps
```

The pattern that emerged is simple:

```bash
uv sync --directory worktrees/fix-builder-pr
uv run --directory worktrees/fix-builder-pr pytest
```

No `cd`, no context switching, no stale environments. The agent (or you)
works in the worktree, and uv finds it by path. When you have five worktrees,
that conciseness compounds.

An alternative is to use a `--bare` clone and manage separate working
directories independently. This avoids git worktree plumbing entirely but
loses the branch-branch safety rails and commands like `git worktree list`.
Both approaches benefit from uv's `--directory` in the same way.

## How the Alternatives Compare

**Conda** has no equivalent. Environments are named (`conda create -n my-env`),
not path-based. You can't run `conda install --directory /some/path`. You'd be
manually activating and deactivating environments or typing full paths into
`--prefix`: both awkward in a multi-worktree setup.

**Pixi** has `--manifest-path` (`-m`). `pixi run --manifest-path
worktrees/my-task/pixi.toml <cmd>` works fine. It's a valid approach for this
workflow.

The difference is speed more than ergonomics. When you're scripting commands
for agents, you write the invocation once in an AGENTS.md or skill file. The
agent doesn't care about a few extra characters. But uv resolves dependencies
faster and runs commands faster. In a workflow where you're spinning up
worktrees repeatedly, that speed compounds.

## Why This Matters

The migration from conda to uv was driven by practical concerns: faster local
development and simpler CI. The worktree parallelism was a discovered benefit,
not a planned one. But it's become the most impactful outcome.

AI coding agents make parallel worktree workflows the default. A migration
decision that made sense on its own terms turned out to position us perfectly
for this shift. That's the kind of accidental foresight worth documenting.
