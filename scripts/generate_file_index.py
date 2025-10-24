#!/usr/bin/env python3
import os, re, ast, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
BEGIN = "<!-- BEGIN: AUTOFILEINDEX -->"
END   = "<!-- END: AUTOFILEINDEX -->"

def short(s, n=120):
    return (s[:n] + "…") if s and len(s) > n else (s or "")

def parse_py(path: pathlib.Path):
    out = {"module_doc": "", "functions": [], "classes": []}
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        return out
    out["module_doc"] = (ast.get_docstring(tree) or "")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out["functions"].append({"name": node.name, "doc": (ast.get_docstring(node) or "").splitlines()[:1]})
        elif isinstance(node, ast.ClassDef):
            out["classes"].append({"name": node.name, "doc": (ast.get_docstring(node) or "").splitlines()[:1]})
    return out

def list_all_files():
    files = []
    for dirpath, _, filenames in os.walk(ROOT):
        parts = pathlib.Path(dirpath).parts
        if any(p in {".venv", ".git", ".idea", "__pycache__"} for p in parts):
            continue
        for fn in sorted(filenames):
            p = pathlib.Path(dirpath) / fn
            rel = p.relative_to(ROOT)
            if rel.name == "generate_file_index.py":
                continue
            files.append(rel)
    return sorted(files, key=lambda p: (str(p).count(os.sep), str(p).lower()))

def build_markdown(files):
    lines = []
    lines += ["", "### Repository files (exact, auto-generated)", "", "> Reflects your working copy at generation time.", ""]
    for rel in files:
        path = ROOT / rel
        if rel.suffix == ".py":
            info = parse_py(path)
            lines.append(f"- `{rel}`")
            if info["module_doc"]:
                lines.append(f"  - *module*: {short(info['module_doc'])}")
            if info["classes"]:
                lines.append("  - **classes**:")
                for c in info["classes"]:
                    doc = short(' '.join(c['doc'])).strip()
                    lines.append(f"    - `{c['name']}` — {doc if doc else 'no docstring'}")
            if info["functions"]:
                lines.append("  - **functions**:")
                for f in info["functions"]:
                    doc = short(' '.join(f['doc'])).strip()
                    lines.append(f"    - `{f['name']}` — {doc if doc else 'no docstring'}")
        else:
            try:
                sz_kb = (path.stat().st_size/1024)
                lines.append(f"- `{rel}` ({sz_kb:.1f} KB)")
            except Exception:
                lines.append(f"- `{rel}`")
    lines.append("")
    return "\n".join(lines)

def rewrite_readme(section_md):
    text = README.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit("Markers not found in README.md")
    new = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END),
                 BEGIN + "\n" + section_md + "\n" + END,
                 text, flags=re.DOTALL)
    README.write_text(new, encoding="utf-8")

if __name__ == "__main__":
    files = list_all_files()
    md = build_markdown(files)
    rewrite_readme(md)
    print("Updated README.md with an exact, per-file index.")
