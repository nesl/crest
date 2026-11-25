"""Render selected repository Markdown sections into the short website guides."""
import re
from pathlib import Path
from urllib.parse import urlsplit

from mkdocs.exceptions import PluginError

INCLUDE = re.compile(r"^(?P<indent> *)<!-- include: (?P<path>[^#\n]+?)#(?P<title>.+?) -->$", re.M)
HEADING = re.compile(r"^(#{1,6}) (.+?)\s*$")
LINK = re.compile(r"(\!?\[[^\]\n]*\]\()([^\s)]+)(\))")


def render_section(source, title, root):
    """Keep subheadings and fenced examples; resolve links against the original file."""
    found = False
    level = 0
    fence = ""
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{%d,} *" % len(fence), line):
                fence = ""
        elif marker:
            fence = marker[1]
        else:
            heading = HEADING.match(line)
            if heading and found and len(heading[1]) <= level:
                break
            if heading and heading[2] == title and not found:
                found, level = True, len(heading[1])
                continue
            if found:
                line = LINK.sub(lambda m: resolve_link(m, source, root), line)
        if found:
            lines.append(line)
    if not found:
        raise PluginError(f"Missing heading {title!r} in {source.relative_to(root)}")
    return "\n".join(lines).strip("\n")


def resolve_link(match, source, root):
    url = urlsplit(match[2])
    if url.scheme or url.netloc or url.path.startswith("/"):
        return match[0]
    target = (source.parent / url.path).resolve() if url.path else source
    kind = "raw" if match[1].startswith("!") else "tree" if target.is_dir() else "blob"
    suffix = ("?" + url.query if url.query else "") + ("#" + url.fragment if url.fragment else "")
    return f"{match[1]}https://github.com/nesl/crest/{kind}/main/{target.relative_to(root).as_posix()}{suffix})"


def on_page_markdown(markdown, *, page, config, files):
    root = Path(config.config_file_path).resolve().parent.parent

    def expand(match):
        text = render_section(root / match["path"], match["title"], root)
        return "\n".join(match["indent"] + line if line else "" for line in text.splitlines())

    # This index contains navigation only. The five API pages use Python docstrings.
    if page.file.src_uri != "api/index.md" and not INCLUDE.search(markdown):
        if not (page.file.src_uri.startswith("api/") and re.search(r"^::: crest\.", markdown, re.M)):
            raise PluginError(
                f"{page.file.src_uri} needs a repository Markdown include or a documented source exception"
            )
    return INCLUDE.sub(expand, markdown)
