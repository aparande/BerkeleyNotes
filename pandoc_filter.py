#!/usr/bin/env python3

import os
import re
import shutil
import sys
from subprocess import call
from tempfile import mkdtemp

from pandocfilters import toJSONFilters
from pandocfilters import Para, Image, get_filename4code, get_extension
from pandocfilters import Math, Space, Str, Header, RawInline, LineBreak, stringify

eqn_labels = []
defn_labels = []
thm_labels = []
table_labels = []
fig_labels = []
section_labels = dict()

unit_map = {
    "degree": "˚", "per": "/", "decade": "dec"
}

def tikz2image(tikz_src, filetype, outfile):
    tmpdir = mkdtemp()
    olddir = os.getcwd()
    shutil.copyfile("gitbook-tikz.tex", tmpdir+"/gitbook-tikz.tex")
    header_file = "gitbook-tikz.tex"
    os.chdir(tmpdir)

    with open('tikz.tex', 'w') as f:
        f.write("\\documentclass[]{standalone}")
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
        call(["convert", tmpdir + '/tikz.pdf', "-density", "500", "-strip", outfile + '.' + filetype])
    shutil.rmtree(tmpdir)

def convert_to_image(format, code):
    outfile = get_filename4code("tikz", code)
    filename = outfile.split("/")[1]
    filetype = get_extension(format, "png", html="png", latex="pdf")
    src = outfile + '.' + filetype
    if not os.path.isfile(src):
        tikz2image(code, filetype, outfile)
        sys.stderr.write('Created image ' + src + '\n')

    return filename, filetype

def convert_figure(format, code):
    """
    Filter to convert tikz pictures into images.
    Taken fom pandocfilters documentation: https://github.com/jgm/pandocfilters/blob/master/examples/tikz.py
    """
    meta_regex = re.compile("(\\\\caption\{(.*?)\})|(\\\\label\{(.*?)\})")
    figure_code = meta_regex.sub("", code)
    figure_code = re.sub("\\\\centering", "", figure_code)
    figure_code = re.sub("(\[H\])|(\[!h\])", "", figure_code)
    figure_code = re.sub("(\\\\begin\{figure\})|(\\\\begin\{table\})", "\\\\begin{minipage}[c]{1.5\\\\linewidth}", figure_code)
    figure_code = re.sub("(figure)|(table)", "minipage", figure_code)
    figure_code = "\n".join(figure_code.split("\n")[1:-1])
    filename, filetype = convert_to_image(format, figure_code)
    
    caption, label = "", None
    for match in meta_regex.finditer(code):
        if match.group(2) is not None:
            caption = match.group(2)
        elif match.group(4) is not None:
            label = match.group(4).split(":")

    if label is not None:
        if label[0] == "fig":
            fig_labels.append(label[1])
            sys.stderr.write(f"Found figure {label[1]}\n")
            caption = f"Figure {len(fig_labels)}: {caption}"
        elif label[0] == "table":
            table_labels.append(label[1])
            sys.stderr.write(f"Found table {label[1]}\n")
            caption = f"Table {len(table_labels)}: {caption}"

    return Para([Image(['', [], []], [Str(caption)], [f"../.gitbook/assets/{filename}.{filetype}", ""])])

def convert_definition(code):
    label_exp = re.compile("(\\\\label\{defn:)(.*?)(\})")
    found_match = False
    for match in label_exp.finditer(code):
        found_match = True
        sys.stderr.write(f"Found definition {match.group(2)}\n")
        defn_labels.append(match.group(2))
        (start, end) = match.span()
        code = code[:start] + code[end:]

    if not found_match:
        defn_labels.append(f"{len(defn_labels)+1}")

    header = Header(3, [f"definition-{len(defn_labels)}", [], []], [Str(f"Definition {len(defn_labels)}")])
    return convert_latex_block(code, header)

def convert_theorem(code):
    label_exp = re.compile("(\\\\label\{thm:)(.*?)(\})")
    found_match = False
    for match in label_exp.finditer(code):
        found_match = True
        sys.stderr.write(f"Found theorem {match.group(2)}\n")
        thm_labels.append(match.group(2))
        (start, end) = match.span()
        code = code[:start] + code[end:]
    if not found_match:
        thm_labels.append(f"{len(thm_labels)+1}")
    header = Header(3, [f"theorem-{len(thm_labels)}", [], []], [Str(f"Theorem {len(thm_labels)}")])
    return convert_latex_block(code, header)


def str_to_math(code):
    code = code.replace("\\[", "$$")
    code = code.replace("\\]", "$$")
    code = code.replace("\\begin{equation}", "$$")
    code = code.replace("\\end{equation}", "$$")
    code = code.replace("\n", "qq")
    
    block = []
    marker = 0
    for match in re.finditer("(\$\$.*?\$\$)|(\\\\cref\{.*?\})|(\$.*?\$)", code):
        (start, end) = match.span()
        if match.group(1) is not None:
            block.append(Str(code[marker:start].replace("qq", "\n")))
            math_value = code[start+2:end-2].replace("qq", " ")

            marker = end
            block.append(Math({"t": "DisplayMath"}, math_value))
        elif match.group(2) is not None:
            block.append(Str(code[marker:start]))
            ref = code[start:end]
            block.append(RawInline("latex", ref))
            marker = end
        elif match.group(3) is not None:
            block.append(Str(code[marker:start].replace("qq", "\n")))
            math_value = code[start+1:end-1].replace("qq", " ")

            marker = end
            block.append(Math({"t": "InlineMath"}, math_value))

    block.append(Str(code[marker:].replace("qq", "\n").strip()))

    return Para(block)

def convert_latex_block(code, header):
    lines = map(lambda x: x.strip(), code.split("\n")[1:-1]) # Remove the \begin and \end
    output = "\n".join(lines)

    output = str_to_math(output)

    hint_tag = Para([Str("{% hint style=\"info\" %}")])
    end_hint_tag = Para([Str("{% endhint %}")])
    return [hint_tag, header, output, end_hint_tag]

def tex_envs(key, value, formt, meta):
    """
    Filter to convert tex environments to markdown
    """
    if key == 'RawBlock':
        [fmt, code] = value
        if fmt == "latex":
            if re.match("\\\\begin{gitbook-image}", code):
                return convert_figure(formt, code)
            elif re.match("\\\\begin{definition}", code):
                return convert_definition(code)
            elif re.match("\\\\begin{theorem}", code):
                return convert_theorem(code)
    elif key == "RawInline":
        [fmt, code] = value
        if fmt == "latex":
            if re.match("\\\\textdegree", code):
                return Str("˚")
    elif key == "Header":
        [level, [label, _, _], content] = value
        section_labels[label] = stringify(content)

def convert_math(code):
    """
    Convert Custom Math Tex code into normal Tex
    """
    diff_exp = re.compile("(\\\\diff\[)(.*?)(\]\{)(.*?)(\}\{)(.*?)(\})")
    code = diff_exp.sub(r"\\frac{d^{\2}\4}{d\6^{\2}}", code)

    vector_bold_exp = re.compile("(\\\\V\{)(.*?)(\})")
    code = vector_bold_exp.sub(r"\\mathbf{\2}", code)

    bold_symbol_exp = re.compile("(\\\\bs\{)(.*?)(\})")
    code = bold_symbol_exp.sub(r"\\boldsymbol{\2}", code)

    sinc_exp = re.compile("\\\\sinc")
    code = sinc_exp.sub(r"\\text{sinc}", code)

    argmin_exp = re.compile("\\\\argmin")
    code = argmin_exp.sub(r"\\text{argmin}", code)

    si_exp = re.compile("(\\\\SI)(\[.*?\])?\{(.*?)\}\{(.*?)\}")
    for match in si_exp.finditer(code):
        num = match.group(3)
        unit = "".join(list(map(lambda x: unit_map.get(x, ''), match.group(4).split("\\"))))
        (start, end) = match.span()
        code = code[:start] + num +"\\text{" + unit + "}" + code[end:]

    code = re.sub("\\\\eqnnumber", " ", code)

    label_exp = re.compile("(\\\\label\{eqn:)(.*?)(\})")
    out = ""
    marker = 0
    for match in label_exp.finditer(code):
        sys.stderr.write(f"Found equation {match.group(2)}\n")
        eqn_labels.append(match.group(2))
        (start, end) = match.span()
        out += code[marker:start] + f"\\qquad ({len(eqn_labels)})"
        marker = end
    out += code[marker:]
    return out.replace("\n", " ")

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
            return [Str("\n\n"), Math(mathType, value), Str("\n\n")]

def label_to_text(label):
    [ref_type, ref_val] = label.split(":")
    if ref_type == "eqn":
        label_num = eqn_labels.index(ref_val) + 1
        return f"equation {label_num}"
    if ref_type == "defn":
        label_num = defn_labels.index(ref_val) + 1
        return f"definition {label_num}"
    if ref_type == "thm":
        label_num = thm_labels.index(ref_val) + 1
        return f"theorem {label_num}"
    if ref_type == "fig":
        label_num = fig_labels.index(ref_val) + 1
        return f"figure {label_num}"
    if ref_type == "table":
        label_num = table_labels.index(ref_val) + 1
        return f"table {label_num}"
    return section_labels.get(label, "(unknown reference)")

def references(key, value, formt, _):
    if key == "RawInline":
        [fmt, code] = value
        if fmt == "latex" and (match := re.match("\\\\cref\{(.*?)\}", code)):
            refs = match.group(1).split(",")
            return Str(", ".join(map(label_to_text, refs)))

def cleanup(key, value, formt, _):
    if key == "RawInline" or key =="RawBlock":
        [fmt, code] = value
        if fmt == "latex":
            if re.match("\\\\cref", code) is None:
                return []
    elif key == "Div":
        [attrs, elems] = value
        return elems

if __name__ == '__main__':
    toJSONFilters([tex_envs, custom_math, references, cleanup])
