"""Embed selected sections of the repository config guide at build time."""
from pathlib import Path
import re
from urllib.parse import urlsplit


def on_page_markdown(markdown, *, page, config, files):
    if '<!-- config-section ' not in markdown:
        return markdown
    source = Path(config.config_file_path).parent.parent / 'src/config/README.md'
    guide = source.read_text(encoding='utf-8')

    def expand(match):
        heading = match[1]
        start = guide.index(f'## {heading}\n') + len(f'## {heading}\n')
        section = guide[start:].split('\n## ', 1)[0].strip()
        def resolve_link(link):
            url = urlsplit(link[1])
            if url.scheme or url.netloc:
                return link[0]
            target = (source.parent / url.path).resolve() if url.path else source.resolve()
            relative = target.relative_to(source.resolve().parents[2])
            fragment = f'#{url.fragment}' if url.fragment else ''
            return f'](https://github.com/nesl/crest/blob/main/{relative.as_posix()}{fragment})'
        section = re.sub(r'\]\(([^)]+)\)', resolve_link, section)
        return '\n'.join('    ' + line if line else '' for line in section.splitlines())

    return re.sub(r'<!-- config-section (.+?) -->', expand, markdown)
