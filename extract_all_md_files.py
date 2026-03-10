import fire
from pathlib import Path

REPO_NAME = "platform-docs-public"
DOCS_PATH = "public"

def list_markdown_paths(include_folders=None, exclude_folders=None, base_path=None) -> list[str]:
    """
    List all .md and .mdx files in public and subfolders.
    Optionally include only certain folders or exclude some folders.
    Args:
        include_folders: list of folder names to include (if set, only these are searched)
        exclude_folders: list of folder names to exclude (these are skipped)
        base_path: base path to the repository (defaults to REPO_NAME)
    """
    if base_path is None:
        base_path = Path(REPO_NAME)
    else:
        base_path = Path(base_path)
    
    docs_dir = base_path / DOCS_PATH
    if not docs_dir.exists():
        print(f"❌ Docs directory not found: {docs_dir}")
        return []
    md_files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.mdx"))
    md_files.sort()
    results = []
    for f in md_files:
        parts = set(f.parts)
        if include_folders and not any(folder in parts for folder in include_folders):
            continue
        if exclude_folders and any(folder in parts for folder in exclude_folders):
            continue
        results.append(str(f))
    return results

if __name__ == "__main__":
    fire.Fire(list_markdown_paths)