Create or edit files inside your CURRENT EDITABLE LAB only (executor-only tool).

The path is resolved against your lab directory and must stay inside it. This tool
can NOT touch the wiki (use wiki_write), can NOT touch the archive (past run dirs
under runs/ are immutable records), and can NOT touch the goal's pinned assets
(e.g. dataset, tokenizer).

Everything in the lab is yours: model, training loop, validation loop, deps in
pyproject.toml, logging. Add metrics when a run needs explaining.
