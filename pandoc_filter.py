#!/usr/bin/env python3

import os
import re
import shutil
import sys
from subprocess import call
from tempfile import mkdtemp

from pandocfilters import toJSONFilters
from pandocfilters import Para, Image, get_filename4code, get_extension
from pandocfilters import Math

def tikz2image(tikz_src, filetype, outfile):
    tmpdir = mkdtemp()
    olddir = os.getcwd()
    shutil.copyfile("custom-tikz.tex", tmpdir+"/custom-tikz.tex")
    header_file = "custom-tikz.tex"
    os.chdir(tmpdir)

    with open('tikz.tex', 'w') as f:
        f.write("\\documentclass{standalone}")
        f.write("\n\\input{" + header_file + "}")
        f.write("""
                \\begin{document}
                """)
        f.write(tikz_src)
        f.write("\n\\end{document}\n")

    call(["pdflatex", 'tikz.tex'], stdout=sys.stderr)
    os.chdir(olddir)
    if filetype == 'pdf':
        shutil.copyfile(tmpdir + '/tikz.pdf', outfile + '.pdf')
    else:
        call(["convert", tmpdir + '/tikz.pdf', "-strip", outfile + '.' + filetype])
    shutil.rmtree(tmpdir)


def tikz(key, value, format, _):
    """
    Filter to convert tikz pictures into images.
    Taken fom pandocfilters documentation: https://github.com/jgm/pandocfilters/blob/master/examples/tikz.py
    """
    if key == 'RawBlock':
        [fmt, code] = value
        if fmt == "latex" and re.match("\\\\begin{tikzpicture}", code):
            outfile = get_filename4code("tikz", code)
            filetype = get_extension(format, "png", html="png", latex="pdf")
            src = outfile + '.' + filetype
            if not os.path.isfile(src):
                tikz2image(code, filetype, outfile)
                sys.stderr.write('Created image ' + src + '\n')
            return Para([Image(['', [], []], [], [src, ""])])

def inline_math(key, value, format, meta):
    """
    GitBooks doesn't recognize inline math delimeters $, so covert it to display math
    """
    if key == 'Math':
        [mathType, value] = value
        if mathType['t'] == "InlineMath":
            mathType['t'] = "DisplayMath"
            return Math(mathType, value)

if __name__ == '__main__':
    toJSONFilters([tikz, inline_math])
