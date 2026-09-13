# DTU Helsinki Asteroid Challenge (HAC) Submission 3

This repostiory contains submission(s) to the Helsinki Asteroid Challenge by the team from the Technical University of Denmark (DTU).

To run this specific submission use this script: submission/submit_inr_convex_prior_mitsuba_stage2_v2_asteroids4to10.sh

The data is too large to comfortably fit within the Git repository.

## Developer Guide
- Implement things that have common use in the library at `src/hac/`
    - Try to keep this relatively neat and use pull requests for any controversial changes
- If you have your own experiments and notebooks, these are best placed in a workspace under `workspaces/`
    - See [below](#contributor-information)
- Add dependencies and manage environment via `uv` to make sure everyone gets working setups
- Run `uv run ruff format .` prior to commiting (or `just format`, see [Development Tasks](#development-tasks))

## Dataset Setup

By default, challenge data is read from `challenge_data` in the project root.

Download the dataset with:

```sh
./bin/get_challenge_data.sh
```

To store the data outside the repository, set `DSDIR` to the directory where `challenge_data` should live:

```sh
export DSDIR="/path/to/data"
./bin/get_challenge_data.sh
```

The script can be run from any directory and will download and extract data into the resolved `challenge_data` directory.

## Development Tasks

[`just`](https://just.systems/) is an optional, convenient task runner for common project commands. All tasks can still be run directly without `just`.

- `just`: List available targets.
- `just format`: Format Python files with Ruff (please run before commiting!).

## Contributor Information

- Add yourself as an author in `pyproject.toml`.
- Create a folder for your own experiments in `workspaces/YOUR_NAME`
- Contribute relevant bits to the common library in `src/hac`
