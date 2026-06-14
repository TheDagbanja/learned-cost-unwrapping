"""Build the manuscript: Markdown -> Word, with native equations, IEEE citations,
and an IEEE two-column layout (single-column title/abstract, two-column body).

    python scripts/build_manuscript.py

Writes manuscript/manuscript.docx (single column) and manuscript/manuscript_2col.docx
(IEEE two-column). Run from the project root so figure paths resolve.
"""

from __future__ import annotations

import shutil
import zipfile

import pypandoc

MD = "manuscript/manuscript.md"
ONE = "manuscript/manuscript.docx"
TWO = "manuscript/manuscript_2col.docx"
PGMAR = ('<w:pgSz w:w="12240" w:h="15840"/>'
         '<w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" '
         'w:header="720" w:footer="720" w:gutter="0"/>')


def main():
    pypandoc.convert_file(
        MD, "docx", outputfile=ONE,
        extra_args=["--citeproc", "--bibliography=manuscript/references.bib",
                    "--csl=manuscript_ieee.csl", "--resource-path=."])
    print("wrote", ONE)

    # Two-column: one-column continuous section for title/abstract, two-column body.
    shutil.copy(ONE, TWO)
    with zipfile.ZipFile(TWO, "r") as z:
        names = z.namelist()
        parts = {n: z.read(n) for n in names}
    xml = parts["word/document.xml"].decode("utf-8")
    i = xml.index("surface deformation.")              # end of the index-terms paragraph
    j = xml.index("</w:p>", i) + len("</w:p>")
    brk = ('<w:p><w:pPr><w:sectPr><w:type w:val="continuous"/>' + PGMAR
           + '<w:cols w:space="720"/></w:sectPr></w:pPr></w:p>')
    xml = xml[:j] + brk + xml[j:]
    k = xml.rfind("</w:sectPr>")                        # body section -> two columns
    xml = xml[:k] + PGMAR + '<w:cols w:num="2" w:space="288"/>' + xml[k:]
    parts["word/document.xml"] = xml.encode("utf-8")
    with zipfile.ZipFile(TWO, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, parts[n])
    print("wrote", TWO, "(IEEE two-column)")


if __name__ == "__main__":
    main()
