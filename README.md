# Berkeley Notes

## Testing Latex to Markdown Conversion

```
pandoc --to json --from latex+raw_tex --output=<target>.json <target-file>

cat <target>.json | ./pandoc_filter.py > <target>-filtered.json

pandoc --to markdown+pipe_tables-simple_tables-smart --from json --atx-headers
--output=<target>.md <target>-filtered.json
```
