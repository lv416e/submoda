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
# Install all tools and Python dependencies
mise install
mise run setup
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
- Commit both files

## Docs

[Complete Specification](docs/specification.md)

## License

Apache-2.0
