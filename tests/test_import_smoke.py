"""
Import smoke tests for installed package usage.
"""


def test_import_smoke():
    import ogeth
    from ogeth import macro_params
    from ogeth.calibrate import demographic_params

    assert ogeth is not None
    assert macro_params is not None
    assert demographic_params is not None
