# Submoda

Submodular optimization platform in Rust.

## What

Large-scale subset selection with provable (1-1/e) approximation guarantee.

- Lazy Greedy, Stochastic Greedy, Continuous Greedy
- Facility Location, Coverage, Log-Determinant
- Thread-safe, deterministic, scalable

## Status

Specification complete. Implementation in progress.

## Development

### Setup

```bash
# Set up the complete development environment
mise run setup
```

### Git Hooks

This project uses [Lefthook](https://github.com/evilmartians/lefthook) for automated code quality checks.

**Installation:**

```bash
# macOS
brew install lefthook

# or using cargo
cargo install lefthook

# Install hooks
lefthook install
```

**Hooks:**
- **pre-commit**: Runs `cargo fmt --check` (~0.3s)
- **pre-push**: Runs `cargo clippy --workspace --all-targets -- -D warnings` (~5s)

**Skip hooks temporarily (for WIP commits):**

```bash
LEFTHOOK=0 git commit -m "WIP: work in progress"
# or
git commit --no-verify -m "WIP: work in progress"
```

**Manual checks** (useful for testing hooks before committing or fixing issues):

```bash
lefthook run pre-commit  # Run formatting check manually
lefthook run pre-push    # Run clippy check manually
```

### Tasks

```bash
mise run test         # Run tests with nextest
mise run check        # Quick compilation check
mise run lint         # Run clippy linter
mise run fmt          # Format code
mise run watch        # Watch and auto-test
mise run bench        # Run benchmarks
mise run pip-compile  # Compile Python dependencies
```

### Python Dependencies

Python dependencies are managed with [pip-tools](https://github.com/jazzband/pip-tools) for full reproducibility:

- Edit `requirements-dev.in` to add/update dependencies
- Run `mise run pip-compile` to generate `requirements-dev.txt`
- Run `mise run setup` to install the updated dependencies
- Commit both files

## Docs

[Complete Specification](docs/specification.md)

## License

Apache-2.0
