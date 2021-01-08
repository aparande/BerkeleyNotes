#!/usr/bin/env python3

import yaml
import sys
import shutil

def copy_files(files, destination):
    for pdffile in files:
        shutil.copy(pdffile['path'], f"{destination}/{pdffile['filename']}")

if __name__ == '__main__':
    destination = sys.argv[1]

    with open("website-files.yml") as f:
        files = yaml.safe_load(f)

    copy_files(files, destination)
