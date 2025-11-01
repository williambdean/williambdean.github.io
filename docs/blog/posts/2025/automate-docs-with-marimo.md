---
tags: 
    - Python
    - Documentation
    - GitHub Actions
comments: true
---

# Automate marimo Notebook Documentation

This is a [follow up](./../2022/site-setup.md) and alternative to using
[`mkdocs`](https://www.mkdocs.org/) for generating documentation: [WASM-powered
HTML via
marimo](https://docs.marimo.io/guides/exporting/?h=expor#export-to-wasm-powered-html)

I initially saw this method of documentation generation on [Vincent
Warmerdam's](https://koaning.io/) python package:
[`mohtml`](https://github.com/koaning/mohtml). Much credit there for the
inspiration.

## Setup

This example requires:

1. Python repository using `uv`: However, you can adapt this to any package manager.
1. [GitHub Pages](https://docs.github.com/en/pages/quickstart) enabled on the repository

## Steps

### Create marimo Notebook

The marimo notebook can be anything you like. In this example, the file is
called `docs.py` and is in the root of the repository you are trying to
generate the documentation for.

Use `marimo edit docs.py` to create or edit the notebook to your liking.

Here is a bare-bones example:

```python
import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    ## My Documentation

    Find all about my project documentation.
    """)
    return


if __name__ == "__main__":
    app.run()
```

### Add GitHub Actions Workflow

Here is the action workflow itself:

```yaml
# .github/workflows/docs.yml
---
name: Documentation
on:
  push:
    branches:
      - main
    paths:
      # The path to your marimo notebook
      - docs.py

jobs:
  create-docs:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      # Checkout the repository
      - uses: actions/checkout@v5

      # Setup uv and install the project
      - name: Install uv
        uses: astral-sh/setup-uv@v7
      - name: Install the project
        run: uv sync --locked --all-extras

      - name: Generate documentation
        run: uv run marimo export html-wasm docs.py -o docs/index.html --mode run

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: {{ '{{' }} secrets.GITHUB_TOKEN {{ '}}' }}
          publish_dir: ./docs
          user_name: 'github-actions[bot]'
          user_email: 'github-actions[bot]@users.noreply.github.com'
```

It will run on every push to main where there was change to `docs.py` file,
generate the HTML, and deploys it to GitHub Pages on the `gh-pages` branch.

### Ensure GitHub Pages is set up correctly

The default deployment branch is `gh-pages`. Ensure that your Pages setup uses this branch. You can check with the following command:

```terminal
gh api /repos/:owner/:repo/pages    
```

Look for `.source.branch` and `.source.path` specifically. The expected output should look like this:


```json
{
  "url": "https://api.github.com/repos/williambdean/frame-search/pages",
  "status": "built",
  "cname": null,
  "custom_404": false,
  "html_url": "https://williambdean.github.io/frame-search/",
  "build_type": "legacy",
  "source": {
    "branch": "gh-pages",
    "path": "/"
  },
  "public": true,
  "protected_domain_state": null,
  "pending_domain_unverified_at": null,
  "https_enforced": true
}
```

You can create in the Settings > Pages section on your repository if need be or
create using the [GitHub
CLI](https://docs.github.com/en/rest/pages/pages?apiVersion=2022-11-28#create-a-github-pages-site).

## Result

The `main` branch will have two new files:

- `docs.py`
- `.github/workflows/docs.yml`

Combined will result in deploying static WASM-powered HTML docs to the `gh-pages` branch.

Link it in your repository and you are good to go!

## More Information

- Check out my repository [`frame-search`](https://github.com/williambdean/frame-search) for a live implementation of this deployment.
- [Gist](https://gist.github.com/williambdean/9abd722feb18abe1c79362aed55ad225) of this GitHub Actions workflow.
- Read an alternative action on the [marimo documentation](https://docs.marimo.io/guides/publishing/github_pages/#publish-using-github-actions)
