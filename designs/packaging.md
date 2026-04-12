# Packaging 📦
## Package management
We are using `uv` with a `pyproject.toml` file to manage the dependencies.

We need to import `blockbuster` as a package in further coding and experiments.

Please don't reinstall things like `pytorch`, `transformers` as we are most probably running on out of box notebook environment. So let's not put them in the dependencies.

## What's in the package
* Shared part code, like model, optimizer, data pipeline, metrics, etc.
* How to make a result dump.
* Where to upload result (usually to a github repo)