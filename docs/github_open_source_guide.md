# Release Checklist

Use this checklist before publishing SafeLoop as a paper code repository.

## Include

- `safety_guard/` runtime package.
- Core scripts for rollout collection, predictor-head training, decision-head training, and compact evaluation.
- Unit tests in `tests/`.
- Public documentation: `README.md`, `MODEL_CARD.md`, this checklist, and the pipeline guide.

## Exclude

- Base policy weights.
- Qwen weights.
- SafeLoop checkpoints.
- Raw rollout images.
- Videos.
- Detailed traces and logs.
- Large JSONL training datasets.
- Machine-specific paths, private notes, credentials, and temporary debug files.

## Required Checks

1. Confirm `.gitignore` excludes outputs, logs, videos, datasets, and checkpoints.
2. Search the files planned for release for private paths and internal notes.
3. Run the unit tests.
4. Inspect `git status --short` and ensure only intended files are staged.
5. Add a license file before public release if one has not already been selected.

## Suggested Git Flow

```bash
git checkout -b release/safeloop
git add README.md MODEL_CARD.md pyproject.toml .gitignore safety_guard scripts tests docs
git status --short
git commit -m "Prepare SafeLoop release repository"
git remote add origin git@github.com:Loule0-0/SafeLoop.git
git push -u origin release/safeloop
```
