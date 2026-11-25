"""Render selected repository Markdown sections into the short website guides."""
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, urlsplit

from mkdocs.exceptions import PluginError

INCLUDE = re.compile(r"^(?P<indent> *)<!-- include: (?P<path>[^#\n]+?)#(?P<title>.+?) -->$", re.M)
HEADING = re.compile(r"^(#{1,6}) (.+?)\s*$")
LINK = re.compile(r"(\!?\[[^\]\n]*\]\()([^\s)]+)(\))")


SOURCE_LABELS = {
    "README.md": "project README",
    "src/README.md": "source code README",
    "src/config/README.md": "config README",
    "src/crest/datasets/README.md": "dataset README",
    "src/crest/tasks/README.md": "task README",
    "src/crest/model_families/README.md": "model-family README",
    "src/crest/microcontrollers/README.md": "hardware backend README",
    "src/GETTING_STARTED.md": "getting-started guide",
    "analysis_scripts/README.md": "analysis scripts README",
}


def on_config(config):
    root = Path(config.config_file_path).resolve().parent.parent
    config.extra["source_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return config


def render_section(source, title, root, revision):
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
                line = LINK.sub(lambda m: resolve_link(m, source, root, revision), line)
        if found:
            lines.append(line)
    if not found:
        raise PluginError(f"Missing heading {title!r} in {source.relative_to(root)}")
    return "\n".join(lines).strip("\n")


def resolve_link(match, source, root, revision):
    url = urlsplit(match[2])
    if url.scheme or url.netloc or url.path.startswith("/"):
        return match[0]
    target = (source.parent / url.path).resolve() if url.path else source
    kind = "raw" if match[1].startswith("!") else "tree" if target.is_dir() else "blob"
    suffix = ("?" + url.query if url.query else "") + ("#" + url.fragment if url.fragment else "")
    return f"{match[1]}https://github.com/nesl/crest/{kind}/{revision}/{target.relative_to(root).as_posix()}{suffix})"


def on_page_markdown(markdown, *, page, config, files):
    root = Path(config.config_file_path).resolve().parent.parent

    revision = config.extra["source_revision"]

    sources = {}

    def expand(match):
        path = match["path"]
        source = root / path
        if not source.is_file():
            raise PluginError(f"{page.file.src_uri} includes missing file {path}")
        try:
            text = render_section(source, match["title"], root, revision)
        except PluginError as error:
            raise PluginError(f"{page.file.src_uri}: {error}") from error
        label = SOURCE_LABELS.get(path, path)
        sources[path] = f"[full {label} on GitHub](https://github.com/nesl/crest/blob/{revision}/{quote(path)})"
        return "\n".join(match["indent"] + line if line else "" for line in text.splitlines())

    # This index contains navigation only. The five API pages use Python docstrings.
    if page.file.src_uri != "api/index.md" and not INCLUDE.search(markdown):
        if not (page.file.src_uri.startswith("api/") and re.search(r"^::: crest\.", markdown, re.M)):
            raise PluginError(
                f"{page.file.src_uri} needs a repository Markdown include or a documented source exception"
            )
    rendered = INCLUDE.sub(expand, markdown)
    if sources:
        links = " and the ".join(sources.values())
        rendered = f"{rendered.rstrip()}\n\nFor more information, read the {links}.\n"
    return rendered
