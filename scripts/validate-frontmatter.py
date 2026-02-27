import re
import sys
from pathlib import Path

import yaml


ALLOWED_TAGS = {
    "Python",
    "Testing",
    "Config Files",
    "Data Analysis",
    "Development",
    "Docker",
    "Pandas",
    "PyMC",
    "Marketing",
    "Design Patterns",
    "Documentation",
    "GitHub Actions",
}

FRONTMATTER_REGEX = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)


def extract_frontmatter(content: str) -> dict | None:
    match = FRONTMATTER_REGEX.match(content)
    if not match:
        return None
    return yaml.safe_load(match.group(1))


def validate_frontmatter(file_path: Path) -> list[str]:
    errors = []
    content = file_path.read_text()
    fm = extract_frontmatter(content)

    if fm is None:
        return ["Missing frontmatter"]

    if "description" not in fm or not fm.get("description"):
        errors.append("Missing or empty 'description'")

    tags = fm.get("tags", [])
    if not tags:
        errors.append("Missing or empty 'tags'")
    elif not isinstance(tags, list):
        errors.append("'tags' must be a list")
    elif len(tags) < 2:
        errors.append("'tags' must have at least 2 items")
    elif len(tags) > 4:
        errors.append("'tags' must have at most 4 items")

    if fm.get("comments") != True:
        errors.append("Missing or invalid 'comments: true'")

    return errors


def main():
    posts_dir = Path("docs/blog/posts")
    errors = []

    for md_file in posts_dir.rglob("*.md"):
        file_errors = validate_frontmatter(md_file)
        if file_errors:
            errors.append(
                f"\n{md_file}:\n" + "\n".join(f"  - {e}" for e in file_errors)
            )

    if errors:
        print("Frontmatter validation failed:")
        print("".join(errors))
        sys.exit(1)
    else:
        print("All blog post frontmatter is valid.")


if __name__ == "__main__":
    main()
