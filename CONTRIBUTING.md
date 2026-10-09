# Contributing

Thanks for your interest. This is an early-stage research tool maintained by one undergraduate developer, so responses may take a few days.

- **Bugs and questions:** open a GitHub issue. Include what you ran, what you expected, what happened, and your Python version. Never attach identifiable or unpublished data.
- **Code changes:** open an issue first to discuss, then a pull request. Every change needs tests, and `pytest` must pass.
- **New input formats or datasets:** add a loader in `src/tracecmr/io.py` that translates into the standard labels in `src/tracecmr/labels.py`, plus a test.

## Support expectations

Issues are reviewed weekly during the academic year. The project follows semantic versioning; results-changing fixes are noted in CHANGELOG.md.
