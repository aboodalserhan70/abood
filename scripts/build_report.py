"""
بناء نسخة HTML جاهزة للطباعة من تقرير Markdown العربي (اتجاه RTL).

الاستخدام:
    python scripts/build_report.py
    python scripts/build_report.py --md report/xxx.md --out report/xxx.html
"""
from __future__ import annotations

import argparse
import base64
import re
from datetime import date
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MD = ROOT / "report" / "corn_leaf_disease_report_ar.md"
DEFAULT_OUT = ROOT / "report" / "corn_leaf_disease_report_ar.html"

CSS = """
:root {
  --ink: #1b2631;
  --muted: #5d6d7e;
  --accent: #1e6f5c;
  --accent-soft: #eaf5f1;
  --line: #dbe4ea;
  --warn-bg: #fff8e6;
  --warn-line: #e0b252;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: #f4f6f7;
  color: var(--ink);
  font-family: "Segoe UI", Tahoma, "Noto Naskh Arabic", "Amiri", "Arial", sans-serif;
  font-size: 16.5px;
  line-height: 1.95;
  direction: rtl;
  text-align: right;
}
.page {
  max-width: 900px;
  margin: 0 auto;
  background: #fff;
  padding: 56px 64px 80px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08), 0 12px 40px rgba(0,0,0,.06);
}
/* ---------- الغلاف ---------- */
.cover {
  border-bottom: 3px double var(--accent);
  padding-bottom: 26px;
  margin-bottom: 16px;
}
.cover .kicker {
  display: inline-block;
  font-size: .78rem;
  letter-spacing: .4px;
  color: var(--accent);
  background: var(--accent-soft);
  border: 1px solid #cfe6de;
  border-radius: 999px;
  padding: 3px 14px;
  margin-bottom: 14px;
}
.cover h1 {
  font-size: 2.1rem;
  line-height: 1.45;
  margin: 0 0 10px;
  color: var(--ink);
  border: 0;
  padding: 0;
}
.cover .subtitle {
  color: var(--muted);
  font-size: 1.02rem;
  margin: 0;
}
/* ---------- العناوين ---------- */
h1, h2, h3, h4 { font-weight: 700; line-height: 1.5; }
h2 {
  font-size: 1.5rem;
  margin: 2.4em 0 .8em;
  padding-bottom: .35em;
  border-bottom: 1px solid var(--line);
  color: #14513f;
  page-break-after: avoid;
}
h2::before {
  content: "";
  display: inline-block;
  width: 10px; height: 10px;
  background: var(--accent);
  border-radius: 3px;
  margin-inline-end: 10px;
  transform: translateY(-2px);
}
h3 {
  font-size: 1.12rem;
  margin: 1.8em 0 .6em;
  color: #1b4f42;
  page-break-after: avoid;
}
hr { border: 0; border-top: 1px solid var(--line); margin: 2.4em 0; }
/* ---------- النص ---------- */
p { margin: .85em 0; text-align: justify; }
strong { color: #0f3d31; }
ul, ol { padding-inline-start: 1.4em; margin: .8em 0; }
li { margin: .35em 0; }
li::marker { color: var(--accent); }
a { color: var(--accent); }
/* ---------- الاقتباسات (تنبيهات) ---------- */
blockquote {
  margin: 1.4em 0;
  padding: 14px 20px;
  background: var(--warn-bg);
  border-inline-start: 4px solid var(--warn-line);
  border-radius: 8px;
  color: #4d3f14;
  font-size: .97rem;
}
blockquote p { margin: .3em 0; }
/* ---------- الجداول ---------- */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5em 0;
  font-size: .93rem;
  page-break-inside: avoid;
}
thead th {
  background: #14513f;
  color: #fff;
  font-weight: 600;
  padding: 10px 12px;
  text-align: right;
  border: 1px solid #14513f;
}
tbody td {
  padding: 9px 12px;
  border: 1px solid var(--line);
  vertical-align: top;
}
tbody tr:nth-child(even) { background: #f8fbfa; }
tbody tr:hover { background: var(--accent-soft); }
table strong { color: #14513f; }
/* جدول البيانات الوصفية في بداية التقرير */
.meta-block table { font-size: .92rem; }
.meta-block thead { display: none; }
.meta-block tbody td:first-child {
  width: 32%;
  font-weight: 700;
  color: #14513f;
  background: #f4f9f7;
}
/* ---------- الأشكال ---------- */
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 1.6em auto .5em;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  page-break-inside: avoid;
}
p > img:only-child { margin-bottom: 0; }
/* ---------- الكود ---------- */
code {
  background: #eef2f4;
  border-radius: 4px;
  padding: .12em .4em;
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: .88em;
  direction: ltr;
  unicode-bidi: embed;
}
pre {
  background: #12232b;
  color: #e6f1f5;
  padding: 16px 18px;
  border-radius: 10px;
  overflow-x: auto;
  direction: ltr;
  text-align: left;
  font-size: .82rem;
  line-height: 1.75;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; color: inherit; font-size: inherit; }
/* ---------- فهرس المحتويات ---------- */
nav.toc {
  background: #f8fbfa;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 18px 26px;
  margin: 28px 0 34px;
  font-size: .95rem;
}
nav.toc h4 {
  margin: 0 0 .6em;
  font-size: .82rem;
  letter-spacing: .6px;
  color: var(--muted);
  text-transform: uppercase;
}
nav.toc ul { list-style: none; padding: 0; margin: 0; }
nav.toc ul ul { padding-inline-start: 1.1em; margin-top: .3em; }
nav.toc li { margin: .3em 0; }
nav.toc a { text-decoration: none; color: #2c3e50; }
nav.toc a:hover { color: var(--accent); text-decoration: underline; }
/* ---------- الطباعة ---------- */
@page { size: A4; margin: 18mm 16mm; }
@media print {
  body { background: #fff; font-size: 11.5pt; }
  .page { box-shadow: none; padding: 0; max-width: none; }
  nav.toc { page-break-after: always; }
  h2 { page-break-before: auto; }
  a { color: inherit; text-decoration: none; }
}
@media (max-width: 720px) {
  .page { padding: 28px 20px 48px; }
  body { font-size: 15.5px; }
  .cover h1 { font-size: 1.6rem; }
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="page">
<header class="cover">
  <span class="kicker">تقرير مشروع · الرؤية الحاسوبية والتعلّم العميق</span>
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>
</header>
{toc}
{body}
</div>
</body>
</html>
"""


def embed_images(html: str, base_dir: Path) -> str:
    """يحوّل الصور إلى data-URI ليصبح ملف HTML مستقلاً تماماً (للطباعة والمشاركة)."""
    def repl(match: re.Match) -> str:
        raw_src = match.group(1)
        img_path = (base_dir / raw_src).resolve()
        if not img_path.exists():
            return match.group(0)
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^":]+?\.(?:png|jpe?g))"', repl, html)


def build(md_path: Path, out_path: Path, embed: bool = True) -> Path:
    text = md_path.read_text(encoding="utf-8")

    # عنوان التقرير والسطر الفرعي
    lines = text.splitlines()
    title = next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), "تقرير")
    subtitle = next((l.lstrip("# ").strip() for l in lines[1:] if l.startswith("## ")), "")
    # نحذف سطر العنوان الرئيسي وسطر العنوان الفرعي من متن التقرير (يُعرضان في الغلاف)
    rest = list(lines[1:]) if lines and lines[0].startswith("# ") else list(lines)
    for i, line in enumerate(rest):
        if not line.strip():
            continue
        if subtitle and line.strip() == f"## {subtitle}":
            rest.pop(i)
        break
    body_md = "\n".join(rest)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "attr_list", "md_in_html",
                    "toc", "sane_lists"],
        extension_configs={"toc": {"toc_depth": "2-3", "anchorlink": False}},
    )
    body = md.convert(body_md)
    toc = md.toc
    # إزالة الحاوية الزائدة التي يولّدها امتداد toc
    toc = toc.removeprefix('<div class="toc">').removesuffix("</div>")

    # التنقل بين العروض
    body = body.replace("<img ", '<img loading="lazy" ')
    # تنظيف المسافات حول الروابط النسبية للصور (تعمل محلياً وعلى GitHub)
    body = re.sub(r'src="figures/', 'src="../figures/', body)

    if embed:
        html = embed_images(
            HTML_TEMPLATE.format(
                title=title,
                subtitle=subtitle or "تصنيف أربع فئات لأوراق الذرة",
                css=CSS,
                toc=f'<nav class="toc"><h4>محتويات التقرير</h4>{toc}</nav>' if toc else "",
                body=body,
            ),
            base_dir=out_path.parent,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        print(f"[تم] {out_path}  ({len(html) / 1024 / 1024:.1f} MB · صور مضمّنة)")
        return out_path

    html = HTML_TEMPLATE.format(
        title=title,
        subtitle=subtitle or "تصنيف أربع فئات لأوراق الذرة",
        css=CSS,
        toc=f'<nav class="toc"><h4>محتويات التقرير</h4>{toc}</nav>' if toc else "",
        body=body,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[تم] {out_path}  ({len(html) / 1024:.0f} KB)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="بناء تقرير HTML عربي من Markdown")
    p.add_argument("--md", type=Path, default=DEFAULT_MD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-embed", action="store_true",
                   help="إبقاء الصور كملفات خارجية بدلاً من تضمينها")
    a = p.parse_args()
    build(a.md, a.out, embed=not a.no_embed)


if __name__ == "__main__":
    main()
