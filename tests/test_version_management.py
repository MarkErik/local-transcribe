"""
Tests for version management system.

Validates that:
- Version is properly exposed in all packages
- Version format is correct
- setuptools-scm is configured correctly
"""

import re
import subprocess
import sys
from pathlib import Path


def test_local_transcribe_version():
    """Test that local_transcribe package exposes __version__."""
    import local_transcribe
    
    assert hasattr(local_transcribe, "__version__"), "local_transcribe should expose __version__"
    assert isinstance(local_transcribe.__version__, str), "__version__ should be a string"
    assert len(local_transcribe.__version__) > 0, "__version__ should not be empty"
    print(f"✓ local_transcribe.__version__ = {local_transcribe.__version__}")


def test_web_api_version():
    """Test that web_api package exposes __version__."""
    # Add parent directory to path to import web_api
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    import web_api
    
    assert hasattr(web_api, "__version__"), "web_api should expose __version__"
    assert isinstance(web_api.__version__, str), "__version__ should be a string"
    assert len(web_api.__version__) > 0, "__version__ should not be empty"
    print(f"✓ web_api.__version__ = {web_api.__version__}")


def test_version_consistency():
    """Test that local_transcribe and web_api have the same version."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    import local_transcribe
    import web_api
    
    assert local_transcribe.__version__ == web_api.__version__, \
        f"Version mismatch: local_transcribe={local_transcribe.__version__}, web_api={web_api.__version__}"
    print(f"✓ Version consistency: {local_transcribe.__version__}")


def test_version_format():
    """Test that version follows expected format."""
    import local_transcribe
    
    version = local_transcribe.__version__
    
    # Should be either:
    # - Clean release: X.Y.Z
    # - Post-release dev: X.Y.postN
    # - Dev fallback: X.Y.Z-dev
    patterns = [
        r'^\d+\.\d+\.\d+$',  # Clean release
        r'^\d+\.\d+\.post\d+$',  # Post-release
        r'^\d+\.\d+\.\d+-dev$',  # Dev fallback
    ]
    
    assert any(re.match(pattern, version) for pattern in patterns), \
        f"Version '{version}' doesn't match expected format"
    print(f"✓ Version format valid: {version}")


def test_version_file_exists():
    """Test that _version.py exists after import."""
    import local_transcribe
    
    version_file = Path(local_transcribe.__file__).parent / "_version.py"
    assert version_file.exists(), f"_version.py should exist at {version_file}"
    print(f"✓ _version.py exists at {version_file}")


def test_version_file_in_gitignore():
    """Test that _version.py is in .gitignore."""
    gitignore = Path(__file__).parent.parent / ".gitignore"
    assert gitignore.exists(), ".gitignore should exist"
    
    content = gitignore.read_text()
    assert "_version.py" in content, "_version.py should be in .gitignore"
    print("✓ _version.py is in .gitignore")


def test_setuptools_scm_config():
    """Test that setuptools-scm is properly configured."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # Fallback for older Python
    
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml should exist"
    
    with open(pyproject, "rb") as f:
        config = tomllib.load(f)
    
    assert "tool" in config, "pyproject.toml should have [tool] section"
    assert "setuptools_scm" in config["tool"], "pyproject.toml should have [tool.setuptools_scm]"
    
    scm_config = config["tool"]["setuptools_scm"]
    assert "write_to" in scm_config, "setuptools_scm should have write_to"
    assert scm_config["write_to"] == "local_transcribe/_version.py", \
        "write_to should point to local_transcribe/_version.py"
    
    print("✓ setuptools-scm is properly configured")


def test_release_script_exists():
    """Test that release.sh exists and is executable."""
    release_script = Path(__file__).parent.parent / "release.sh"
    assert release_script.exists(), "release.sh should exist"
    assert release_script.stat().st_mode & 0o111, "release.sh should be executable"
    print("✓ release.sh exists and is executable")


def test_release_script_syntax():
    """Test that release.sh has valid bash syntax."""
    release_script = Path(__file__).parent.parent / "release.sh"
    result = subprocess.run(
        ["bash", "-n", str(release_script)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"release.sh has syntax errors: {result.stderr}"
    print("✓ release.sh has valid syntax")


if __name__ == "__main__":
    # Run all tests
    tests = [
        test_local_transcribe_version,
        test_web_api_version,
        test_version_consistency,
        test_version_format,
        test_version_file_exists,
        test_version_file_in_gitignore,
        test_setuptools_scm_config,
        test_release_script_exists,
        test_release_script_syntax,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            failed.append((test.__name__, str(e)))
            print(f"✗ {test.__name__}: {e}")
    
    print()
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed!")
        sys.exit(0)
