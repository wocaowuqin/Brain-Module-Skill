# Translation And Extraction Notes

- Source read: lawful open-access arXiv version `2111.02209`, submitted 3 November 2021.
- Published record: IEEE TNSM 19(3), 3276-3289, 2022, DOI `10.1109/TNSM.2022.3159670`.
- The arXiv LaTeX source and original figure PDFs were used because PDF text extraction from the two-column layout produced encoding and reading-order errors.
- All fourteen source figures were converted directly from the original figure PDFs. No full-page screenshot is used as a figure substitute.
- Equations are preserved at the conceptual/formula level in `paper.md`; the complete notation and derivations remain in `source.pdf` and `source_tex/NFVAI.tex`.
- The related-work discussion is consolidated rather than translated citation by citation. The complete references are retained in `source_tex/NFVAI.bbl`.
- The paper's abstract reports admission improvement of 7%-14%, while its conclusion reports 7%-20%. This inconsistency is preserved and flagged.
- `tolerable latency` is translated as `可容忍时延`; it denotes SFC/core-network latency in this paper, not the full application E2E SLA measured by this project's UDP probes.
- Reader status is `full-structure draft`: all substantive sections, algorithms, parameters, results and figures are represented, but it is not a sentence-by-sentence translation of the bibliography or every related-work sentence.
