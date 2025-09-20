def pytest_addoption(parser) -> None:
    parser.addoption("--confidence", type=float, default=0.5)
    parser.addoption("--time-limit", type=float, default=60)
