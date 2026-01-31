# Version Management

This project uses [setuptools-scm](https://github.com/pypa/setuptools-scm) for automatic version management based on git tags.

## How It Works

- **Development versions**: When working on untagged commits, the version includes `.dev{N}` (e.g., `0.1.dev767`)
- **Release versions**: Tagged commits get clean semantic versions (e.g., `0.1.0`, `1.2.3`)
- **setuptools-scm** automatically generates version from git tags and commits
- **_version.py** is auto-generated and should not be committed to version control

## Version Sources

The version is dynamically generated and available in multiple places:

- `pyproject.toml`: Project version (used by `uv`, pip, etc.)
- `web_api/__init__.py`: API version (used by the web server)
- `local_transcribe/_version.py`: Auto-generated version file

## Creating a Release

Use the `release.sh` script to create new releases:

```bash
# Patch release (0.1.0 -> 0.1.1)
./release.sh

# Minor release (0.1.0 -> 0.2.0)
./release.sh minor

# Major release (0.1.0 -> 1.0.0)
./release.sh major

# Specific version
./release.sh 2.0.0
```

The script will:
1. Validate the current version and workspace state
2. Check if the tag already exists
3. Update `web_ui/package.json` to match the new version
4. Commit the web_ui version change
5. Create an annotated git tag
6. Build the package to verify everything works
7. Optionally push both commits and tag to remote
8. Rollback changes automatically if build fails

## Version Format

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Checking Current Version

```bash
# Check API version
uv run python -c "from web_api import __version__; print(__version__)"

# Check project version
uv run python -c "import local_transcribe; print(local_transcribe.__version__)"
```Ensure working directory is clean (no uncommitted changes)
3. When ready for release, run `./release.sh` with appropriate bump type
4. The script will:
   - Update web_ui version to match
   - Create a commit for the version bump
   - Tag the commit with the new version
   - Build the package for verification
5. Review and push the release to remote
6# Development Workflow
Python package, API, and web UI
- **Safe**: Automatic rollback on build failures
- **Validated**: Format checking and duplicate tag prevention

## Configuration Details

### setuptools-scm Settings

In `pyproject.toml`:

```toml
[tool.setuptools_scm]
write_to = "local_transcribe/_version.py"
version_scheme = "post-release"
local_scheme = "no-local-version"
```

- `version_scheme = "post-release"`: Uses post-release versioning (0.1.0.post1)
- `local_scheme = "no-local-version"`: Cleaner version strings without local identifiers in releases
- `write_to`: Auto-generates `_version.py` (excluded from git)

### Version Exposure

The version is exposed in three places:

1. **Python package**: `local_transcribe.__version__`
2. **Web API**: `web_api.__version__` (imports from local_transcribe)
3. **Web UI**: `web_ui/package.json` (manually synced by release.sh)

## Troubleshooting

### _version.py not generated

Run `uv build` to trigger setuptools-scm generation.

### Version stuck on old value

Make sure you've created and checked out the git tag:
```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git checkout v1.0.0
```

### Build fails during release

The script automatically rolls back:
- Deletes the created tag
- Reverts the web_ui package.json commit
- Leaves your repository in the pre-release state
1. Make changes and commit them
2. When ready for release, run `./release.sh` with appropriate bump type
3. The version will automatically update for tagged commits
4. Deploy the tagged version to production

## Benefits

- **Automatic**: No manual version file updates needed
- **Accurate**: Version reflects exact git state
- **Traceable**: Each release is tied to a specific commit
- **Consistent**: Same version across all components</content>
<parameter name="filePath">/Users/ai/ai-Dev/local-transcribe/docs/VERSION_MANAGEMENT.md