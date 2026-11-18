import optix


def test_version_present() -> None:
    assert isinstance(optix.__version__, str)
    assert optix.__version__
