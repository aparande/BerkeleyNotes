# Berkeley Notes
While taking courses at the University of California, Berkeley, I have created detailed notes for many of the classes I have taken.
This repository contains those notes in PDF format and the LaTeX source. The
PDFs are posted on [my website](anmolparande.com/resources) and in [online
format](notes.anmolparande.com).

For any typos or fixes, please raise a GitHub issue or create a pull request.

## Notes
The notes in this repository are
- EE16B: Designing Information Devices and Systems II
- EE120: Signals and Systems
- EE123: Digital Signal Processing
- EECS126: Probability and Random Processes
- EECS127: Optimization Models in Engineering
- EE128: Feedback Control
- EECS225A: Statistical Signal Processing
- CS61B: Data Structures
- CS61C: Machine Structures

## Testing LaTeX to Markdown Conversion

The online notes are hosted on GitBook, which requires Markdown. The markdown
formatted notes are in the `gitbook` branch. To automatically convert from LaTeX
to markdown, I use [pandoc](https://pandoc.org). Pandoc can't do the conversion
directly, so the `pandoc_filter.py` script applies a filter to the AST that
pandoc creates to modify it for proper conversion. The actual filters are
written using my
[latex-pandoc-filters](https://github.com/aparande/latex-pandoc-filters)
repository. The actual filters used are defined in `pandoc.yaml`.

```
pandoc --to json --from latex+raw_tex --output=<target>.json <target-file>

cat <target>.json | ./pandoc_filter.py > <target>-filtered.json

pandoc --to markdown+pipe_tables-simple_tables-smart --from json --atx-headers
--output=<target>.md <target>-filtered.json
```

After all the notes are converted into Markdown, each class is in a single
document. The last step for formatting on GitBook is to split each note into one
note for each section. That is done using `create_book.py`. This script also
creates the `SUMMARY.md` which GitBook uses as the title page.

## GitHub Workflows

When the LaTeX source is updated, the GitBook and the website notes are
automatically updated through GitHub workflows. The `site-deployment` workflow
takes the PDFs from the repository and pushes them to the website repository. It
is run when a commit is pushed to `main`.  The PDFs which are pushed are listed
in `website-files.yml` and are copied using `website-copy.py`.  The `gitbook`
workflow converts the LaTeX into Markdown and pushes it to the gitbook branch.
It is run when a commit is pushed to `main`. The `gitbook-test` workflow
converts the LaTeX into Markdown and produces a diff on the gitbook branch as
well as the Markdown as artifacts for review. It is run on pull-request.
