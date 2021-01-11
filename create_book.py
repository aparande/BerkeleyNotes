#!/usr/bin/env python3

import sys
import re
import os
import shutil


def break_file(filepath):
    with open(filepath, 'r') as f:
        contents = f.read()

    filename = os.path.basename(filepath).split(".")[0]

    new_dir = filename
    try:
        shutil.rmtree(new_dir)
    except:
        pass

    os.mkdir(new_dir)

    marker = 0
    titles = ["Introduction"]
    header_map = []
    for idx, match in enumerate(re.finditer("^#\s(.*?)(\n|\{)", contents, flags=re.M)):
        [start, end] = match.span()
        header_title = match.group(1)
        titles.append(header_title)

        new_file = f"{new_dir}/{filename}-{idx}.md"
        with open(new_file, "w") as f:
            f.write(contents[marker:start])
            marker = start

        header_map.append((new_file, titles[idx]))


    new_file = f"{new_dir}/{filename}-{len(titles)-1}.md"
    with open(new_file, "w") as f:
        f.write(contents[marker:])

    header_map.append((new_file, titles[-1]))

    print(header_map)
    return header_map

if __name__ == '__main__':
    input_files = sys.argv[1:]

    maps = map(lambda x: (x, break_file(x)), input_files)

    with open("SUMMARY.md", "w") as f:
        f.writelines(["# Table of contents\n", "* [Introduction](README.md)\n"])
        for input_file, header_map in maps:
            dirname = os.path.basename(input_file).split(".")[0]
            f.writelines(map(lambda x: f"\t* [{x[1]}]({x[0]})\n", header_map))
            os.remove(input_file)

