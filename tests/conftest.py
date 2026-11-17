def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full soliton fits (about a minute with numba, much longer without)")

def pytest_collection_modifyitems(config, items):
    import pytest
    if "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow; run with: pytest -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
