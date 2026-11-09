<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Contributing

Bug reports and focused pull requests are welcome.
Include the configuration, reproduction steps, software versions, and target board when reporting a problem.
Remove credentials and private data before sharing logs or configurations.
For component work, start with the [source guide](src/README.md#where-to-start), then follow the component's examples, extension steps, and completion checks.

## Development environment

Follow [Getting started](src/GETTING_STARTED.md).
The supported setup installs dependencies from `environment.yml`, then installs CREST with `make install`, which uses `pip install -e . --no-deps`.
`pyproject.toml` declares package dependencies for Python packaging.
Neither file is a complete environment lock.
When adding a dependency, update both declarations as appropriate and explain the requirement in the pull request.

## Tests

Run from the repository root with the environment active.

```bash
make test
```

The default suite needs no connected board or live LLM credentials.
It excludes integration tests and three opt-in analysis test modules.
To include those analysis tests, run the following command.

```bash
RUN_ANALYSIS_SCRIPT_TESTS=1 make test
```

Run toolchain integration checks separately after installing their prerequisites.
These include Arduino compile-only checks, which need the CLI and board packages but no connected board.

```bash
make integration-test
```

For backend changes, also follow the [hardware validation checklist](src/crest/microcontrollers/README.md#definition-of-done) on the target board.
Software tests alone do not validate physical measurements.

## Change guidelines

- Keep changes focused and cover changed behavior with tests.
- Preserve public imports and configuration compatibility, or document intentional breaking changes and migration steps.
- Use NumPy-style docstrings for changed functions and classes, and explain non-obvious decisions with inline comments.
- Keep hardware logic testable through configuration, staging, request, and parser checks before connected-device validation.
- Update runnable examples and extension instructions when their usage changes.
  Keep implementation detail beside the code rather than repeating it across READMEs.
- Use one sentence per line in Markdown prose and preserve code-block formatting.
