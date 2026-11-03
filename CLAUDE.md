# roblox-thumbgen — agent instructions

## Second brain: `notes/` (Obsidian vault)

`notes/` is the project's memory. At the start of every session read `notes/INDEX.md`, then the
latest file in `notes/log/`. During the session:

- Record any decision (model, hyper-parameter, caption format, API shape) as a dated note in
  `notes/decisions/` and link it from `notes/decisions/INDEX.md`.
- Put research findings and training results in `notes/research/`.
- Keep `notes/pipeline/*.md` in sync with the code when you change a stage.
- End the session with `notes/log/YYYY-MM-DD-session-N.md` (what changed, what is next) and link it from `notes/log/INDEX.md` and the "Start here" list in `notes/INDEX.md`.

Note conventions: YAML frontmatter (`title`, `date`, `tags` list), relative markdown links (no wikilinks),
a `Related:` line at the end, distinct basenames. Never write `.obsidian/workspace.json` into the repo.
Lint with the obsidian-notes skill before committing notes.

## Code

- Package: `thumbgen/` (CLI `thumbgen ingest|serve|prompt`). Configs in `configs/`. No GPU in the cloud
  session: test prompt building, ingest filtering and the API with CPU-only unit tests in `tests/`.
- Generated outputs, datasets and model weights go under `work/` (git-ignored).
