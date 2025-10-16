# Claude Code Spec-Driven Development

Kiro-style Spec Driven Development implementation using claude code slash commands, hooks and agents.

## Project Context

### Project Information
- **Project Name**: `submoda` (repository name)
- **Crate Prefix**: `submod-` (e.g., submod-core, submod-solver, submod-io)
- **License**: Apache-2.0

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`
- Commands: `.claude/commands/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- **submoda**: Production-grade submodular optimization platform for large-scale subset selection problems
  - Phase: ready-for-implementation
  - Status: Requirements ✅ | Design ✅ | Tasks ✅
  - Next: `/kiro:spec-impl submoda` to begin implementation

Use `/kiro:spec-status [feature-name]` to check detailed progress.

## Development Guidelines
- Think in English, generate responses in English

### Conventional Commits Format

Enforce the following format for both commit messages and PR titles:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Important**: PR titles must also follow the `<type>(<scope>): <description>` format

**Type List**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code formatting
- `refactor`: Refactoring
- `test`: Test addition/modification
- `chore`: Build, configuration, maintenance
- `ci`: CI/CD configuration
- `perf`: Performance improvement

**Scope Examples (This Project)**
- `dev`: Development environment
- `core`: submod-core
- `solver`: submod-solver
- `io`: submod-io
- `objectives`: submod-objectives
- `docs`: Documentation

## Workflow

### Phase 0: Steering (Optional)
`/kiro:steering` - Create/update steering documents
`/kiro:steering-custom` - Create custom steering for specialized contexts

Note: Optional for new features or small additions. You can proceed directly to spec-init.

### Phase 1: Specification Creation
1. `/kiro:spec-init [detailed description]` - Initialize spec with detailed project description
2. `/kiro:spec-requirements [feature]` - Generate requirements document
3. `/kiro:spec-design [feature]` - Interactive: "Have you reviewed requirements.md? [y/N]"
4. `/kiro:spec-tasks [feature]` - Interactive: Confirms both requirements and design review

### Phase 2: Progress Tracking
`/kiro:spec-status [feature]` - Check current progress and phases

## Development Rules
1. **Consider steering**: Run `/kiro:steering` before major development (optional for new features)
2. **Follow 3-phase approval workflow**: Requirements → Design → Tasks → Implementation
3. **Approval required**: Each phase requires human review (interactive prompt or manual)
4. **No skipping phases**: Design requires approved requirements; Tasks require approved design
5. **Update task status**: Mark tasks as completed when working on them
6. **Keep steering current**: Run `/kiro:steering` after significant changes
7. **Check spec compliance**: Use `/kiro:spec-status` to verify alignment

## Git Hooks

This project uses [Lefthook](https://github.com/evilmartians/lefthook) for automated code quality checks.

**Why Lefthook?**
- Fast and language-agnostic (written in Go)
- Supports parallel execution of hooks
- Simple YAML configuration
- Cross-platform compatibility (macOS, Linux, Windows)
- No runtime dependencies (single binary)
- Better performance compared to alternatives like Husky (Node.js) or pre-commit (Python)

### Setup
Developers must install lefthook after cloning:
```bash
brew install lefthook  # or: cargo install lefthook
lefthook install
```

### Hook Behavior
- **pre-commit**: Validates code formatting with `cargo fmt --check` (fast, ~0.3s)
- **pre-push**: Runs clippy lints with `cargo clippy --workspace --all-targets -- -D warnings` (thorough, ~5s)

### Bypassing Hooks
For work-in-progress commits, you may skip hooks:
```bash
LEFTHOOK=0 git commit -m "WIP: feature in progress"
# or
git commit --no-verify -m "WIP: feature in progress"
```

**Important**: Final commits should always pass all hooks before creating a PR.

### Philosophy
- Hooks catch simple mistakes locally (formatting, obvious errors)
- They provide fast feedback before push
- They complement (not replace) CI and CodeRabbit reviews
- Developers can bypass for WIP, but production code must pass

## Steering Configuration

### Current Steering Files
Managed by `/kiro:steering` command. Updates here reflect command changes.

### Active Steering Files
- `product.md`: Always included - Product context and business objectives
- `tech.md`: Always included - Technology stack and architectural decisions
- `structure.md`: Always included - File organization and code patterns

### Custom Steering Files
<!-- Added by /kiro:steering-custom command -->
<!-- Format:
- `filename.md`: Mode - Pattern(s) - Description
  Mode: Always|Conditional|Manual
  Pattern: File patterns for Conditional mode
-->

### Inclusion Modes
- **Always**: Loaded in every interaction (default)
- **Conditional**: Loaded for specific file patterns (e.g., "*.test.js")
- **Manual**: Reference with `@filename.md` syntax
