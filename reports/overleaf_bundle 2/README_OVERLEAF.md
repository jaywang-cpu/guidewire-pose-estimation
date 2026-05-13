# Overleaf bundle — IEEE conference paper format

A self-contained LaTeX project for the EN.580.627 final report, formatted
as an **IEEE Conference Paper** (`IEEEtran` document class, two-column,
Times New Roman font via `newtxtext`/`newtxmath`).

## Contents

| File | Purpose |
|---|---|
| `main.tex` | The full paper (Title / Abstract / I–V + References) |
| `refs.bib` | BibTeX bibliography (16 entries) |
| `figures/` | 19 PNG figures (5 referenced by `\includegraphics` in `main.tex`) |
| `README_OVERLEAF.md` | This file |

## Open in Overleaf (30 seconds)

1. Compress this folder into one `.zip` (already done as
   `reports/overleaf_bundle.zip` in the repo).
2. Sign in to <https://www.overleaf.com>.
3. **New Project → Upload Project** → select the zip.
4. Overleaf will unpack the project. Set the **Main document** to
   `main.tex` if it isn't already, and the **Compiler** to **pdfLaTeX**.
5. Click **Recompile**. The first compile will run BibTeX automatically
   (Overleaf detects the `\bibliography{}` call); citations resolve from
   `refs.bib`.

## What is in `main.tex`

- `\documentclass[conference,10pt,letterpaper]{IEEEtran}` — IEEEtran is
  pre-installed on Overleaf, no extra upload needed.
- `\usepackage{newtxtext}` + `\usepackage{newtxmath}` — sets **Times New
  Roman** for both body text and math (the IEEE-paper standard).
- Two-column layout with proper IEEE conference paper styling: Abstract,
  Index Terms, numbered sections (I, II, …), subsections (A, B, …), single
  bibliography column.
- 5 figures embedded as `\begin{figure}…\end{figure}` floats with proper
  `\caption{}` and `\label{}`; cross-referenced via `\ref{}`.
- 6 tables embedded as `\begin{table}…\end{table}` floats with the
  `booktabs` package.
- 1 algorithm pseudocode block (`algorithmic` package).
- 16 citations via `\cite{…}` resolved against `refs.bib` with the
  `IEEEtran` bibliography style.

## How to switch to a different template

If you prefer NeurIPS, ACM, Elsevier, etc., keep everything between
`\begin{document}` and `\end{document}` and replace the preamble at the top.
The body uses only generic LaTeX commands (`\section`, `\includegraphics`,
`\cite`, `\begin{table}`, `\begin{algorithm}`) and will compile under any
common template after minor preamble adjustments.

To change the font away from Times New Roman, remove or replace the
`newtxtext` / `newtxmath` lines:

```latex
% Remove these for default LM/CM:
\usepackage{newtxtext}
\usepackage{newtxmath}
```

## Push to an existing Overleaf project via git (optional)

```bash
cd reports/overleaf_bundle/
git init -b main
git add main.tex refs.bib figures/ README_OVERLEAF.md
git commit -m "Import guidewire-pose final paper"
# Get the project's git URL from Overleaf:
#   Menu (top-left) → Git → "Clone with HTTPS"
git remote add overleaf https://git.overleaf.com/<PROJECT_ID>
git push -u overleaf main
```

(Overleaf will prompt for your account email and a Git authentication
token, which you generate at Account Settings → Git Integration.)
