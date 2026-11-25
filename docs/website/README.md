# Website Link

https://nesl.github.io/crest/

## Acknowledgments

Parts of this project page were adapted from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), which was adapted from the [Nerfies](https://nerfies.github.io) project page.

## Website License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

## Updating the documentation

The guides in `docs/guide/` select sections from repository Markdown files.
For example, the model-family guide includes the following section at build time:

```markdown
<!-- include: src/crest/model_families/README.md#Add A New Model Family -->
```

Edit the source section to update both GitHub and the website. The section includes
its subheadings. If you rename the heading, update its include directive in the same
PR. A missing heading fails the site build. The API reference reads Python docstrings
through mkdocstrings. The documentation landing page and navigation are authored separately.

Install the packages in `docs/requirements.txt` and run `make site` from the repository
root to check the result. GitHub Actions runs this build on pull requests and
publishes successful builds from main when Pages is configured for Actions.

### Documentation sources and exceptions

| Pages | Where to edit | Reason |
| --- | --- | --- |
| Datasets, tasks, model families, boards | The extension section in each component README under `src/crest/` | Contributors read and edit the same instructions on GitHub and the website. |
| Configuration overview and four settings guides | `src/config/README.md` | The short examples and full field descriptions belong with the shipped YAML files. |
| Trial concepts, measurement windows, extension overview | `src/README.md` and `src/crest/microcontrollers/README.md` | These describe the shared pipeline and the backend measurement contract. The site adds diagrams and navigation. |
| Run overview, first study, reproduction | Getting-started guide, root, analysis and source READMEs | These pages select the existing workflow or analysis instructions. |
| Setup, hardware, results | Root and source READMEs plus `src/GETTING_STARTED.md` | The getting-started file already owns the operator commands. Including it directly keeps those commands in one place. |
| Documentation home | Root README for supported workloads and hardware, site template for the interactive trial map | The clickable regions, keyboard labels and guide destinations belong to the website. |
| Registry, dataset, task, model-family and device API references | Python classes, functions and NumPy-style docstrings | mkdocstrings reads the actual signatures and contracts at build time. |
| API reference index | `docs/guide/api/index.md` | It routes readers to the generated API pages and contains no implementation procedure. |

The BSD footer serves the root `LICENSE`. The third-party notice is maintained in
`docs/guide/legal/third-party-notices.txt` because it identifies the documentation
toolchain and its bundled assets. The project homepage under `docs/website/` is
a separate paper and software project page outside the MkDocs guide.

New guide pages need a repository Markdown include. For a new exception, record
the source and reason here and update the build check in `docs/hooks/readme_sections.py`.

For a source archive without Git metadata, set `CREST_SOURCE_REVISION` to the release tag or commit when building. Otherwise source links from archive builds open `main`.
