#!/usr/bin/env python3

import os
import re
import shutil
import sys
from subprocess import call
from tempfile import mkdtemp

from pandocfilters import toJSONFilters
from pandocfilters import Para, Image, get_filename4code, get_extension
from pandocfilters import Math, Space, Str, Header

global definition_num
definition_num = 0

global theorem_num
theorem_num = 0

def tikz2image(tikz_src, filetype, outfile):
    tmpdir = mkdtemp()
    olddir = os.getcwd()
    shutil.copyfile("gitbook-tikz.tex", tmpdir+"/gitbook-tikz.tex")
    header_file = "gitbook-tikz.tex"
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

def convert_tikz(format, code):
    """
    Filter to convert tikz pictures into images.
    Taken fom pandocfilters documentation: https://github.com/jgm/pandocfilters/blob/master/examples/tikz.py
    """
    outfile = get_filename4code("tikz", code)
    filename = outfile.split("/")[1]
    filetype = get_extension(format, "png", html="png", latex="pdf")
    src = outfile + '.' + filetype
    if not os.path.isfile(src):
        tikz2image(code, filetype, outfile)
        sys.stderr.write('Created image ' + src + '\n')
    return Para([Image(['', [], []], [], [f".gitbook/assets/{filename}.{filetype}", ""])])

def convert_definition(code):
    global definition_num
    definition_num += 1
    header = Header(3, [f"definition-{definition_num}", [], []], [Str(f"Definition {definition_num}")])
    return convert_latex_block(code, header)

def convert_theorem(code):
    global theorem_num
    theorem_num += 1
    header = Header(3, [f"theorem-{theorem_num}", [], []], [Str(f"Theorem {theorem_num}")])
    return convert_latex_block(code, header)

def str_to_math(code):
    code = code.replace("$", "$$")
    code = code.replace("\\[", "\n$$")
    code = code.replace("\\]", "$$\n")

    block = []
    marker = 0
    for match in re.finditer("\$\$.*\$\$", code):
        (start, end) = match.span()
        block.append(Str(code[marker:start]))
        math_value = code[start+2:end-2]
        marker = end + 1
        block.append(Math({"t": "DisplayMath"}, math_value))

    block.append(Str(code[marker:]))

    return Para(block)

def convert_latex_block(code, header):
    lines = code.split("\n")[1:-1] # Remove the \begin and \end
    output = "\n".join(lines)

    output = str_to_math(output)

    hint_tag = Para([Str("{% hint style=\"info\" %}")])
    end_hint_tag = Para([Str("{% endhint %}")])
    return [hint_tag, header, output, end_hint_tag]

def tex_envs(key, value, formt, _):
    if key == 'RawBlock':
        [fmt, code] = value
        if fmt == "latex":
            if re.match("\\\\begin{tikzpicture}", code):
                return convert_tikz(formt, code)
            elif re.match("\\\\begin{tabularx}", code):
                return convert_tikz(formt, code)
            elif re.match("\\\\begin{definition}", code):
                return convert_definition(code)
            elif re.match("\\\\begin{theorem}", code):
                return convert_theorem(code)
    elif key == "RawInline":
        [fmt, code] = value
        if fmt == "latex":
            if re.match("\\\\textdegree", code):
                return Str("˚")

def convert_math(code):
    diff_exp = re.compile("(\\\\diff\[)(.*?)(\]\{)(.*?)(\}\{)(.*?)(\})")
    code = diff_exp.sub(r"\\frac{d^{\2}\4}{d\6^{\2}}", code)

    vector_bold_exp = re.compile("(\\\\V\{)(.*?)(\})")
    code = vector_bold_exp.sub(r"\\mathbf{\2}", code)

    bold_symbol_exp = re.compile("(\\\\bs\{)(.*?)(\})")
    code = bold_symbol_exp.sub(r"\\boldsymbol{\2}", code)
    return code

def custom_math(key, value, format, meta):
    """
    GitBooks doesn't recognize inline math delimeters $, so covert it to display math
    """
    if key == 'Math':
        [mathType, value] = value
        value = convert_math(value)
        if mathType['t'] == "InlineMath":
            mathType['t'] = "DisplayMath"
            return Math(mathType, value)
        else:
            return Math(mathType, value)

if __name__ == '__main__':
    toJSONFilters([tex_envs, custom_math])
