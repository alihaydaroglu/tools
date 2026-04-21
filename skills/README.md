# Tools skills

Claude Code skills bundled with the `tools` package. Each skill is a directory containing a `SKILL.md` with frontmatter (`name`, `description`) plus a body of instructions. Claude Code loads them on demand based on the `description` field.

## Skills in this directory

- **plotting-conventions** — house rules for plots (shared bins, identity lines, colormaps, three-format saves).
- **pca-analysis** — which PCA module to use and which cross-validation method fits the data.

More will be added as analysis-specific guidance accumulates (NMF, UMAP, CCA, etc.).

## How Claude Code discovers skills

Claude Code looks for skills in two locations:

- `~/.claude/skills/<skill-name>/SKILL.md` — user-level, available in any project
- `.claude/skills/<skill-name>/SKILL.md` — project-level, only in that project

Skills stored here in the tools repo are **not** discovered automatically. They have to be symlinked (or copied) into one of those locations.

## Installation

From the tools repo root:

```bash
make install-skills
```

Symlinks every directory under `tools/skills/` into `~/.claude/skills/`. Editing a `SKILL.md` in this repo immediately affects what Claude Code sees.

To remove: `make uninstall-skills`. To list: `make list-skills`.

## Authoring notes

- Skill names are lowercase-hyphenated and must match the directory name.
- The `description` field is how Claude Code decides relevance — be specific about triggers.
- Skills should reference concrete utilities in `tools/` (e.g. `tools.plot.scatter`, `tools.pca.PCA`) so they guide users toward the canonical implementations.
