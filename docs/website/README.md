# Website Link

https://nesl.github.io/crest/

## Acknowledgments

Parts of this project page were adopted from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), which was adopted from the [Nerfies](https://nerfies.github.io) project page.

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
