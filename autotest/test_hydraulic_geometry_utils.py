import numpy as np
import pytest

import pywatershed as pws
from pywatershed.base.parameters import Parameters
from pywatershed.hydrology.prms_hydraulic_geometry import CFS_TO_CMS

NSEG = 3
NHRU = 3


def _meta(dims: tuple, units: str) -> dict:
    return {"dims": dims, "attrs": {"units": units}}


@pytest.fixture
def synthetic_params() -> Parameters:
    dims = {"nsegment": NSEG, "nhru": NHRU}
    coords = {
        "nhm_seg": np.array([101, 102, 103], dtype=np.int64),
        "nhm_id": np.array([1, 2, 3], dtype=np.int64),
    }
    data_vars = {
        "tosegment": np.array([3, 3, 0], dtype=np.int64),
        "tosegment_nhm": np.array([103, 103, 0], dtype=np.int64),
        "seg_length": np.array([1000.0, 2000.0, 1500.0]),
        "seg_slope": np.array([0.01, 0.005, 0.002]),
        "mann_n": np.array([0.04, 0.035, 0.03]),
        "seg_width": np.array([5.0, 8.0, 12.0]),
        "seg_depth": np.array([0.5, 0.8, 1.2]),
        "hru_segment": np.array([1, 2, 3], dtype=np.int64),
        "hru_elev": np.array([120.0, 110.0, 100.0]),
    }
    metadata = {
        "global": {},
        "nhm_seg": _meta(("nsegment",), "none"),
        "nhm_id": _meta(("nhru",), "none"),
        "tosegment": _meta(("nsegment",), "none"),
        "tosegment_nhm": _meta(("nsegment",), "none"),
        "seg_length": _meta(("nsegment",), "meters"),
        "seg_slope": _meta(("nsegment",), "decimal fraction"),
        "mann_n": _meta(("nsegment",), "seconds / meter ** (1/3)"),
        "seg_width": _meta(("nsegment",), "meter"),
        "seg_depth": _meta(("nsegment",), "meter"),
        "hru_segment": _meta(("nhru",), "none"),
        "hru_elev": _meta(("nhru",), "meters"),
    }
    return Parameters(
        dims=dims, coords=coords, data_vars=data_vars, metadata=metadata
    )


def _manning_bankfull(w, d, s, n):
    area = w * d
    radius = area / (w + 2 * d)
    return area * radius ** (2.0 / 3.0) * np.sqrt(s) / n


@pytest.mark.domainless
def test_at_a_station_hand_computed(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    new, bankfull = at_a_station_hydraulic_geometry(
        synthetic_params, return_bankfull=True
    )
    p = synthetic_params.parameters
    q_bf = _manning_bankfull(
        p["seg_width"], p["seg_depth"], p["seg_slope"], p["mann_n"]
    )
    np.testing.assert_allclose(bankfull["bankfull_flow"], q_bf, rtol=1e-12)
    # one literal anchor so the test does not only mirror the formula:
    # segment 0, w=5, d=0.5, s=0.01, n=0.04 -> area 2.5, R 0.41667
    np.testing.assert_allclose(q_bf[0], 3.486629948344633, rtol=1e-12)
    np.testing.assert_allclose(
        bankfull["bankfull_velocity"],
        q_bf / (p["seg_width"] * p["seg_depth"]),
        rtol=1e-12,
    )
    assert bankfull["velocity_exp"] == pytest.approx(0.34)

    newp = new.parameters
    np.testing.assert_allclose(newp["width_m"], 0.26)
    np.testing.assert_allclose(newp["depth_m"], 0.40)
    np.testing.assert_allclose(
        newp["width_alpha"], p["seg_width"] / q_bf**0.26, rtol=1e-12
    )
    np.testing.assert_allclose(
        newp["depth_alpha"], p["seg_depth"] / q_bf**0.40, rtol=1e-12
    )
    # the new parameters carry metadata on the segment dimension; the
    # units are the registry's (parameters.yaml), which says "unknown" for
    # width_m but "none" for depth_m
    for name, units in (
        ("width_alpha", "unknown"),
        ("width_m", "unknown"),
        ("depth_alpha", "meters"),
        ("depth_m", "none"),
    ):
        assert new.metadata[name]["dims"] == ("nsegment",)
        assert new.metadata[name]["attrs"]["units"] == units
    # the other parameters are carried over untouched
    np.testing.assert_array_equal(newp["seg_length"], p["seg_length"])
    assert new.dims["nsegment"] == NSEG


@pytest.mark.domainless
def test_at_a_station_exponent_override(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    new, bankfull = at_a_station_hydraulic_geometry(
        synthetic_params, width_exp=0.1, depth_exp=0.5, return_bankfull=True
    )
    np.testing.assert_allclose(new.parameters["width_m"], 0.1)
    np.testing.assert_allclose(new.parameters["depth_m"], 0.5)
    assert bankfull["velocity_exp"] == pytest.approx(0.4)


@pytest.mark.domainless
def test_at_a_station_returns_parameters_only_by_default(
    synthetic_params,
):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    new = at_a_station_hydraulic_geometry(synthetic_params)
    assert isinstance(new, Parameters)
    assert "depth_alpha" in new.parameters


@pytest.mark.domainless
def test_at_a_station_does_not_mutate_input(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    _ = at_a_station_hydraulic_geometry(synthetic_params)
    assert "depth_alpha" not in synthetic_params.parameters
    assert "width_alpha" not in synthetic_params.parameters


@pytest.mark.domainless
def test_at_a_station_overwrites_existing_geometry(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["width_alpha"] = np.full(NSEG, 99.0)
    dd.metadata["width_alpha"] = _meta(("nsegment",), "unknown")
    dd.data_vars["width_m"] = np.full(NSEG, 0.015)
    dd.metadata["width_m"] = _meta(("nsegment",), "none")
    with_old = Parameters(**dd.data)

    new = at_a_station_hydraulic_geometry(with_old)
    assert not np.any(new.parameters["width_alpha"] == 99.0)
    np.testing.assert_allclose(new.parameters["width_m"], 0.26)


@pytest.mark.domainless
def test_at_a_station_slope_floor(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )
    from pywatershed.utils.network_hydraulics import SLOPE_FLOOR

    dd = synthetic_params.to_dd()
    dd.data_vars["seg_slope"][:] = 0.0
    flat = Parameters(**dd.data)
    _, bankfull = at_a_station_hydraulic_geometry(flat, return_bankfull=True)
    p = flat.parameters
    expected = _manning_bankfull(
        p["seg_width"], p["seg_depth"], SLOPE_FLOOR, p["mann_n"]
    )
    np.testing.assert_allclose(bankfull["bankfull_flow"], expected)


@pytest.mark.domainless
@pytest.mark.parametrize("name", ["seg_width", "seg_depth", "mann_n"])
def test_at_a_station_nonpositive_raises(synthetic_params, name):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars[name][1] = 0.0
    bad = Parameters(**dd.data)
    with pytest.raises(ValueError, match=f"{name}.*1 segment"):
        at_a_station_hydraulic_geometry(bad)


@pytest.mark.domainless
def test_at_a_station_nan_slope_raises(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["seg_slope"][1] = np.nan
    bad = Parameters(**dd.data)
    with pytest.raises(ValueError, match="seg_slope.*1 segment"):
        at_a_station_hydraulic_geometry(bad)


@pytest.mark.domainless
def test_at_a_station_negative_slope_raises(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["seg_slope"][1] = -0.01
    bad = Parameters(**dd.data)
    with pytest.raises(ValueError, match="seg_slope.*1 segment"):
        at_a_station_hydraulic_geometry(bad)


@pytest.mark.domainless
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"width_exp": np.nan}, "width_exp must be in"),
        ({"depth_exp": -0.1}, "depth_exp must be in"),
        ({"width_exp": 1.2}, "width_exp must be in"),
        # velocity exponent 1 - 0.6 - 0.6 would be negative
        ({"width_exp": 0.6, "depth_exp": 0.6}, "must not exceed 1"),
    ],
)
def test_at_a_station_bad_exponents_raise(synthetic_params, kwargs, match):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    with pytest.raises(ValueError, match=match):
        at_a_station_hydraulic_geometry(synthetic_params, **kwargs)


@pytest.mark.domainless
def test_at_a_station_missing_raises(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    del dd.data_vars["seg_depth"]
    del dd.metadata["seg_depth"]
    missing = Parameters(**dd.data)
    with pytest.raises(ValueError, match="seg_depth"):
        at_a_station_hydraulic_geometry(missing)


@pytest.mark.domainless
def test_at_a_station_drb_through_process(synthetic_params):
    """The derived parameters, fed to PRMSHydraulicGeometryFull's own
    calculation with the bankfull flow in cfs (the process input unit),
    reproduce the bankfull width and depth. This fails if either side
    changes its flow unit or parameter names; the algebraic round trip
    alpha * q**m == w cannot."""
    import types

    from pywatershed.hydrology.prms_hydraulic_geometry import (
        PRMSHydraulicGeometryFull,
    )
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    param_file = (
        pws.constants.__pywatershed_root__ / "data/drb_2yr/myparam.param"
    )
    params = pws.parameters.PrmsParameters.load(param_file)
    new, bankfull = at_a_station_hydraulic_geometry(
        params, return_bankfull=True
    )
    assert isinstance(new, pws.parameters.PrmsParameters)
    needed = set(PRMSHydraulicGeometryFull.get_parameters())
    assert needed <= set(new.parameters)
    p = params.parameters
    nseg = p["seg_width"].size
    q_cms = bankfull["bankfull_flow"]
    assert np.all(q_cms > 0)

    # stand in for a process instance: its parameters plus the process's
    # input (cfs) and output arrays
    fake = types.SimpleNamespace(
        **{name: new.parameters[name] for name in needed},
        seg_outflow=q_cms / CFS_TO_CMS,
        seg_flow_width=np.zeros(nseg),
        seg_flow_depth=np.zeros(nseg),
        seg_flow_area=np.zeros(nseg),
        seg_flow_velocity=np.zeros(nseg),
        seg_res_time=np.zeros(nseg),
    )
    PRMSHydraulicGeometryFull._compute_hydraulic_geometry(fake)
    np.testing.assert_allclose(fake.seg_flow_width, p["seg_width"], rtol=1e-10)
    np.testing.assert_allclose(fake.seg_flow_depth, p["seg_depth"], rtol=1e-10)
    np.testing.assert_allclose(
        fake.seg_flow_velocity, bankfull["bankfull_velocity"], rtol=1e-10
    )
