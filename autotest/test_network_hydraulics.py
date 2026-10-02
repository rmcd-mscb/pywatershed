import pathlib as pl
import types
import warnings

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString

import pywatershed as pws
from pywatershed.base.parameters import Parameters

# Three-reach synthetic network: reaches 0 and 1 are headwaters that
# flow into reach 2, which is the outlet.
NSEG = 3
NHRU = 3


def _meta(dims: tuple, units: str) -> dict:
    return {"dims": dims, "attrs": {"units": units}}


@pytest.fixture
def synthetic_params() -> Parameters:
    dims = {"nsegment": NSEG, "nhru": NHRU, "scalar": 1}
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
        "elev_units": np.array([1], dtype=np.int64),
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
        "hru_elev": _meta(("nhru",), "elev_units"),
        "elev_units": _meta(("scalar",), "none"),
    }
    return Parameters(
        dims=dims, coords=coords, data_vars=data_vars, metadata=metadata
    )


@pytest.mark.domainless
def test_shear_velocity_values():
    from pywatershed.utils.network_hydraulics import G, shear_velocity

    depth = np.array([1.0, 2.0, 0.0])
    slope = np.array([0.001, 0.0, 0.01])
    result = shear_velocity(depth, slope)
    expected = np.array(
        [np.sqrt(G * 1.0 * 0.001), np.sqrt(G * 2.0 * 1.0e-7), 0.0]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.domainless
def test_calculate_seg_mid_elevations(synthetic_params):
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    mid, outlet_mid = calculate_seg_mid_elevations(synthetic_params)
    np.testing.assert_allclose(mid, np.array([108.0, 108.0, 101.5]))
    assert outlet_mid == {2: 101.5}


def _with_elev_units(params: Parameters, value) -> Parameters:
    """Copy of ``params`` with ``elev_units`` replaced (or removed if None)."""
    dd = params.to_dd()
    if value is None:
        del dd.data_vars["elev_units"]
        del dd.metadata["elev_units"]
    else:
        dd.data_vars["elev_units"] = np.array([value], dtype=np.int64)
    return Parameters(**dd.data)


@pytest.mark.domainless
def test_calculate_seg_mid_elevations_feet(synthetic_params):
    """elev_units=0: hru_elev is feet, converted; seg_dy stays in meters."""
    from pywatershed.constants import meters_per_foot
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    mid, outlet_mid = calculate_seg_mid_elevations(
        _with_elev_units(synthetic_params, 0)
    )
    # outlet HRU elevation 100 ft = 30.48 m; rises are 10, 10, 3 m
    outlet_m = 100.0 * meters_per_foot
    expected = np.array(
        [outlet_m + 3.0 + 5.0, outlet_m + 3.0 + 5.0, outlet_m + 1.5]
    )
    np.testing.assert_allclose(mid, expected)
    assert list(outlet_mid) == [2]
    np.testing.assert_allclose(outlet_mid[2], outlet_m + 1.5)


def _with_hru_segment(params: Parameters, hru_segment) -> Parameters:
    dd = params.to_dd()
    dd.data_vars["hru_segment"] = np.array(hru_segment, dtype=np.int64)
    return Parameters(**dd.data)


@pytest.mark.domainless
def test_calculate_seg_mid_elevations_outlet_without_hru(synthetic_params):
    """An outlet with no HRU takes its datum from the nearest upstream
    HRUs, less the rise of the segments between: HRUs at 110 and 100 m
    drain to reach 1, whose downstream end is the outlet's upstream end,
    so the outlet's downstream end is 100 - 3 = 97 m."""
    from pywatershed.utils.mmr_to_mf6_dfw import MmrToMf6Dfw
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    params = _with_hru_segment(synthetic_params, [1, 2, 2])
    with pytest.warns(UserWarning, match="Outlet segment index 2 has no"):
        mid, outlet_mid = calculate_seg_mid_elevations(params)
    np.testing.assert_allclose(mid, np.array([105.0, 105.0, 98.5]))
    np.testing.assert_allclose(outlet_mid[2], 98.5)

    # the debug check applies the same rule
    fake = types.SimpleNamespace(parameters=params)
    with pytest.warns(UserWarning, match="Outlet segment index 2 has no"):
        MmrToMf6Dfw._calculate_seg_mid_elevations(fake, check=True)
    np.testing.assert_allclose(fake._seg_mid_elevation, mid)


@pytest.mark.domainless
def test_calculate_seg_mid_elevations_outlet_takes_lowest_hru(
    synthetic_params,
):
    """Two HRUs on the outlet at 110 and 100 m: the datum is the lower."""
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["hru_segment"] = np.array([1, 3, 3], dtype=np.int64)
    dd.data_vars["hru_elev"] = np.array([120.0, 110.0, 100.0])
    mid, outlet_mid = calculate_seg_mid_elevations(Parameters(**dd.data))
    np.testing.assert_allclose(mid, np.array([108.0, 108.0, 101.5]))
    assert outlet_mid == {2: 101.5}


@pytest.mark.domainless
def test_calculate_seg_mid_elevations_drb_pinned():
    """Values recorded from MmrToMf6Dfw._calculate_seg_mid_elevations on
    develop before the refactor (2026-10-02), so the walk is pinned on a
    real 456-segment network, not only the 3-reach synthetic one."""
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    params = pws.parameters.PrmsParameters.load(
        pws.constants.__pywatershed_root__ / "data/drb_2yr/myparam.param"
    )
    mid, outlet_mid = calculate_seg_mid_elevations(params)
    np.testing.assert_allclose(mid.min(), 0.35908267095, rtol=1e-12)
    np.testing.assert_allclose(mid.max(), 1038.6766781532647, rtol=1e-12)
    np.testing.assert_allclose(mid.sum(), 75865.88027301495, rtol=1e-12)
    expected_outlets = {
        24: 0.35908267095,
        94: 2.34790010425,
        95: 1.0702298366,
        102: 1.0895265485475,
        104: 1.32095499435,
        194: 1.5386658365,
    }
    assert sorted(outlet_mid) == sorted(expected_outlets)
    for kk, vv in expected_outlets.items():
        np.testing.assert_allclose(outlet_mid[kk], vv, rtol=1e-12)


@pytest.mark.domainless
def test_validate_tosegment_non_integer_raises():
    from pywatershed.utils.network_hydraulics import _validate_tosegment

    with pytest.raises(ValueError, match="tosegment must be 0"):
        _validate_tosegment(np.array([3.5, 3.0, 0.0]))


@pytest.mark.domainless
def test_calculate_seg_mid_elevations_no_hru_anywhere_raises(
    synthetic_params,
):
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    # hru_segment = 0 is PRMS for "not routed to any segment"
    params = _with_hru_segment(synthetic_params, [0, 0, 0])
    with pytest.raises(ValueError, match="to any segment upstream"):
        calculate_seg_mid_elevations(params)


@pytest.mark.domainless
@pytest.mark.parametrize(
    "value,match",
    [(None, "elev_units is required"), (2, "must be the scalar 0")],
)
def test_calculate_seg_mid_elevations_elev_units_raises(
    synthetic_params, value, match
):
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    with pytest.raises(ValueError, match=match):
        calculate_seg_mid_elevations(_with_elev_units(synthetic_params, value))


@pytest.mark.domainless
def test_calculate_seg_mid_elevations_two_outlets(synthetic_params):
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["tosegment"] = np.array([3, 0, 0], dtype=np.int64)
    two_outlets = Parameters(**dd.data)

    mid, outlet_mid = calculate_seg_mid_elevations(two_outlets)
    np.testing.assert_allclose(mid, np.array([108.0, 115.0, 101.5]))
    assert outlet_mid == {1: 115.0, 2: 101.5}


NTIME = 4
TIMES = np.arange(
    np.datetime64("1979-01-01"), np.datetime64("1979-01-05")
).astype("datetime64[ns]")


def _write_run_var(run_dir, name, values, units, nhm_seg):
    da = xr.DataArray(
        values,
        dims=("time", "nhm_seg"),
        coords={"time": TIMES, "nhm_seg": nhm_seg},
        name=name,
        attrs={"units": units},
    )
    da.to_netcdf(run_dir / f"{name}.nc")


@pytest.fixture
def synthetic_run_dir(tmp_path, synthetic_params) -> pl.Path:
    """A fake pywatershed output directory for the three-reach network."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nhm_seg = synthetic_params.parameters["nhm_seg"]
    base = np.array([[10.0, 20.0, 35.0]])  # cfs, outlet sums the two
    ramp = np.arange(1, NTIME + 1)[:, None]  # 1..4
    outflow = base * ramp
    inflow = outflow * 0.9
    width = np.array([[4.0, 6.0, 10.0]]) * np.ones((NTIME, 1))
    depth = np.array([[0.3, 0.5, 0.9]]) * ramp * 0.5
    velocity = outflow * 0.028316847 / (width * depth)
    res_time = (
        width
        * depth
        * synthetic_params.parameters["seg_length"]
        / (outflow * 0.028316847)
    )
    _write_run_var(run_dir, "seg_outflow", outflow, "cfs", nhm_seg)
    _write_run_var(run_dir, "seg_inflow", inflow, "cfs", nhm_seg)
    _write_run_var(run_dir, "seg_flow_width", width, "meters", nhm_seg)
    _write_run_var(run_dir, "seg_flow_depth", depth, "meters", nhm_seg)
    _write_run_var(
        run_dir, "seg_flow_velocity", velocity, "meters per second", nhm_seg
    )
    _write_run_var(run_dir, "seg_res_time", res_time, "seconds", nhm_seg)
    return run_dir


@pytest.mark.domainless
def test_export_static_fields(synthetic_params, synthetic_run_dir, tmp_path):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    out = export_network_hydraulics(
        synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
    )
    assert out == tmp_path / "net.nc"
    ds = xr.open_dataset(out)
    p = synthetic_params.parameters
    assert ds.sizes["reach"] == NSEG
    assert ds.sizes["time"] == NTIME
    np.testing.assert_array_equal(ds["reach_id"], p["nhm_seg"])
    np.testing.assert_array_equal(ds["to_id"], p["tosegment_nhm"])
    np.testing.assert_array_equal(ds["to_index"], np.array([2, 2, -1]))
    np.testing.assert_array_equal(ds["is_outlet"], np.array([0, 0, 1]))
    assert ds["to_index"].dtype == np.int32
    assert ds["is_outlet"].dtype == np.int8
    assert ds["reach_id"].dtype == np.int64
    np.testing.assert_array_equal(ds["length"], p["seg_length"])
    np.testing.assert_array_equal(ds["slope"], p["seg_slope"])
    np.testing.assert_array_equal(ds["mann_n"], p["mann_n"])
    np.testing.assert_array_equal(ds["bankfull_width"], p["seg_width"])
    np.testing.assert_array_equal(ds["bankfull_depth"], p["seg_depth"])
    np.testing.assert_allclose(
        ds["elevation_mid"], np.array([108.0, 108.0, 101.5])
    )
    assert "vertex" not in ds.dims
    assert "x_mid" not in ds
    assert ds["length"].attrs["units"] == "m"
    assert ds["to_index"].attrs["source_name"] == "tosegment"
    assert ds.attrs["source_model"] == "pywatershed PRMS"
    assert "pywatershed_version" in ds.attrs
    assert ds.attrs["n_unconnected"] == -1  # no polyline supplied
    ds.close()


@pytest.mark.domainless
def test_export_static_fields_to_id_derived_without_tosegment_nhm(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    dd = synthetic_params.to_dd()
    del dd.data_vars["tosegment_nhm"]
    del dd.metadata["tosegment_nhm"]
    no_tosegment_nhm = Parameters(**dd.data)

    out = export_network_hydraulics(
        no_tosegment_nhm, synthetic_run_dir, tmp_path / "net.nc"
    )
    ds = xr.open_dataset(out)
    np.testing.assert_array_equal(ds["to_id"], np.array([103, 103, 0]))
    ds.close()


@pytest.mark.domainless
def test_export_time_varying_fields(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.hydrology.prms_hydraulic_geometry import CFS_TO_CMS
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
        shear_velocity,
    )

    out = export_network_hydraulics(
        synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
    )
    ds = xr.open_dataset(out)
    src = {
        nm: xr.open_dataarray(synthetic_run_dir / f"{nm}.nc").load()
        for nm in [
            "seg_outflow",
            "seg_inflow",
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_velocity",
            "seg_res_time",
        ]
    }
    np.testing.assert_array_equal(ds["time"], TIMES)
    np.testing.assert_allclose(
        ds["flow_out"], src["seg_outflow"].values * CFS_TO_CMS
    )
    np.testing.assert_allclose(
        ds["flow_in"], src["seg_inflow"].values * CFS_TO_CMS
    )
    np.testing.assert_allclose(ds["width"], src["seg_flow_width"].values)
    np.testing.assert_allclose(ds["depth"], src["seg_flow_depth"].values)
    np.testing.assert_allclose(ds["velocity"], src["seg_flow_velocity"].values)
    np.testing.assert_allclose(
        ds["residence_time"], src["seg_res_time"].values
    )
    expected_ustar = shear_velocity(
        src["seg_flow_depth"].values,
        synthetic_params.parameters["seg_slope"][None, :],
    )
    np.testing.assert_allclose(ds["ustar"], expected_ustar)
    # literal anchor: depth[0, 0] = 0.15 m, slope[0] = 0.01
    np.testing.assert_allclose(
        ds["ustar"].values[0, 0], 0.12128468576040423, rtol=1e-12
    )
    assert ds["flow_out"].attrs["units"] == "m3 s-1"
    assert ds["flow_out"].attrs["source_name"] == "seg_outflow"
    assert ds["ustar"].attrs["method"] == "sqrt(g*depth*max(slope, 1e-7))"
    assert ds["velocity"].attrs["method"] == "power_law_at_a_station"
    assert ds["flow_out"].dims == ("time", "reach")
    assert "water_temperature" not in ds
    ds.close()


@pytest.mark.domainless
def test_export_optional_temperature(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    temp = np.full((NTIME, NSEG), 12.5)
    _write_run_var(
        synthetic_run_dir,
        "seg_tave_water",
        temp,
        "degrees Celsius",
        synthetic_params.parameters["nhm_seg"],
    )
    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
    )
    np.testing.assert_allclose(ds["water_temperature"], temp)
    assert ds["water_temperature"].attrs["units"] == "degC"
    ds.close()


@pytest.mark.domainless
def test_export_time_subset(synthetic_params, synthetic_run_dir, tmp_path):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            start_time=np.datetime64("1979-01-02"),
            end_time=np.datetime64("1979-01-03"),
        )
    )
    assert ds.sizes["time"] == 2
    np.testing.assert_array_equal(ds["time"], TIMES[1:3])
    ds.close()


@pytest.mark.domainless
def test_export_missing_files_raise(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    (synthetic_run_dir / "seg_flow_depth.nc").unlink()
    (synthetic_run_dir / "seg_res_time.nc").unlink()
    with pytest.raises(FileNotFoundError) as excinfo:
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
    assert "seg_flow_depth" in str(excinfo.value)
    assert "seg_res_time" in str(excinfo.value)


@pytest.mark.domainless
def test_export_reach_order_mismatch_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    path = synthetic_run_dir / "seg_outflow.nc"
    with xr.open_dataarray(path) as opened:
        da = opened.load()  # close the file before rewriting it (Windows)
    da = da.assign_coords(nhm_seg=np.array([103, 102, 101]))
    da.to_netcdf(path)
    with pytest.raises(ValueError, match="nhm_seg"):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )


def _write_segments_shp(path, lines, ids, crs="EPSG:5070"):
    gdf = gpd.GeoDataFrame(
        {"nsegment_v": ids, "model_idx": np.arange(1, len(ids) + 1)},
        geometry=[LineString(ll) for ll in lines],
        crs=crs,
    )
    gdf.to_file(path)


@pytest.fixture
def synthetic_lines():
    # reach 0: two-vertex line ending at the junction (0, 0)
    # reach 1: three-vertex line, digitized BACKWARDS (starts at junction)
    # reach 2: outlet, from the junction to (0, -1500)
    return [
        [(-1000.0, 0.0), (0.0, 0.0)],
        [(0.0, 0.0), (500.0, 1000.0), (1000.0, 2000.0)],
        [(0.0, 0.0), (0.0, -1500.0)],
    ]


@pytest.mark.domainless
def test_export_polyline_block(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    # shuffle the shapefile row order to prove matching is by id
    _write_segments_shp(
        shp,
        [synthetic_lines[2], synthetic_lines[0], synthetic_lines[1]],
        [103, 101, 102],
    )
    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )
    )
    assert ds.attrs["n_unconnected"] == 0
    assert "5070" in ds.attrs["crs_wkt"] and "Albers" in ds.attrs["crs_wkt"]
    np.testing.assert_array_equal(ds["reach_vertex_count"], [2, 3, 2])
    np.testing.assert_array_equal(ds["reach_vertex_start"], [0, 2, 5])
    assert ds.sizes["vertex"] == 7
    vx = ds["vertex_x"].values
    vy = ds["vertex_y"].values
    vd = ds["vertex_dist"].values
    # reach 0 as digitized
    np.testing.assert_allclose(vx[0:2], [-1000.0, 0.0])
    np.testing.assert_allclose(vd[0:2], [0.0, 1000.0])
    # reach 1 was reversed so that it ends at the junction
    np.testing.assert_allclose(vx[2:5], [1000.0, 500.0, 0.0])
    np.testing.assert_allclose(vy[2:5], [2000.0, 1000.0, 0.0])
    seg = np.hypot(500.0, 1000.0)
    np.testing.assert_allclose(vd[2:5], [0.0, seg, 2 * seg])
    # reach 2 (outlet) untouched
    np.testing.assert_allclose(vy[5:7], [0.0, -1500.0])
    np.testing.assert_allclose(vd[5:7], [0.0, 1500.0])
    # midpoints at half arc length
    np.testing.assert_allclose(ds["x_mid"], [-500.0, 500.0, 0.0])
    np.testing.assert_allclose(ds["y_mid"], [0.0, 1000.0, -750.0])
    assert ds["vertex_dist"].attrs["units"] == "m"
    ds.close()


@pytest.mark.domainless
def test_export_polyline_backwards_outlet_reversed(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    lines = [list(ll) for ll in synthetic_lines]
    lines[2] = [(0.0, -1500.0), (0.0, 0.0)]  # outlet digitized backwards
    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, lines, [101, 102, 103])
    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )
    )
    vy = ds["vertex_y"].values
    vd = ds["vertex_dist"].values
    np.testing.assert_allclose(vy[5:7], [0.0, -1500.0])
    np.testing.assert_allclose(vd[5:7], [0.0, 1500.0])
    np.testing.assert_allclose(ds["y_mid"][2], -750.0)
    assert ds.attrs["n_unconnected"] == 0
    ds.close()


@pytest.mark.domainless
def test_export_polyline_unconnected_counted(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    lines = [list(ll) for ll in synthetic_lines]
    lines[0] = [(-1000.0, 50.0), (0.0, 50.0)]  # displaced by 50 m
    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, lines, [101, 102, 103])
    with pytest.warns(UserWarning, match="1 reach polyline"):
        out = export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )
    ds = xr.open_dataset(out)
    assert ds.attrs["n_unconnected"] == 1
    ds.close()


@pytest.mark.domainless
@pytest.mark.parametrize(
    "line,match",
    [
        ([(-1000.0, 0.0), (-1000.0, 0.0)], "Reach 101 polyline has zero"),
        ([], "Reach 101 has a null or empty"),
    ],
)
def test_export_polyline_degenerate_line_raises(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path, line, match
):
    """A particle position is s/length scaled onto the polyline, so a
    zero-length or empty line cannot be used and must name the reach."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    lines = [list(ll) for ll in synthetic_lines]
    lines[0] = line
    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, lines, [101, 102, 103])
    with pytest.raises(ValueError, match=match):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )


@pytest.mark.domainless
def test_export_polyline_connect_tol_is_used(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    """The 50 m gap that counts as unconnected at the default 1 m is
    connected at connect_tol=100, and the value is recorded."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    lines = [list(ll) for ll in synthetic_lines]
    lines[0] = [(-1000.0, 50.0), (0.0, 50.0)]  # displaced by 50 m
    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, lines, [101, 102, 103])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
            connect_tol=100.0,
        )
    ds = xr.open_dataset(out)
    assert ds.attrs["n_unconnected"] == 0
    assert ds.attrs["connect_tol"] == 100.0
    ds.close()


@pytest.mark.domainless
def test_export_polyline_missing_id_column_raises(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, synthetic_lines, [101, 102, 103])
    with pytest.raises(ValueError, match="Column nope not in"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
            shp_id_col="nope",
        )


@pytest.mark.domainless
def test_export_polyline_fractional_ids_raise(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, synthetic_lines, [101.4, 102.0, 103.0])
    with pytest.raises(ValueError, match="integer identifiers"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )


@pytest.mark.domainless
def test_export_transposed_run_file_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    """A (nhm_seg, time) file must be rejected by name, not transposed or
    left to xarray's shape error."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    path = synthetic_run_dir / "seg_inflow.nc"
    with xr.open_dataarray(path) as opened:
        da = opened.load()
    da.transpose("nhm_seg", "time").to_netcdf(path)
    with pytest.raises(ValueError, match=r"seg_inflow.nc has dims"):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )


@pytest.mark.domainless
def test_export_polyline_id_mismatch_raises(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, synthetic_lines, [101, 102, 999])
    with pytest.raises(ValueError, match="nsegment_v"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )


@pytest.mark.domainless
def test_export_polyline_geographic_crs_raises(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, synthetic_lines, [101, 102, 103], crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )


@pytest.mark.domainless
def test_export_polyline_non_meter_crs_raises(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    # NAD83 / Pennsylvania South (US survey feet)
    _write_segments_shp(shp, synthetic_lines, [101, 102, 103], crs="EPSG:2272")
    with pytest.raises(ValueError, match="meter"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )


@pytest.mark.domainless
def test_export_polyline_missing_crs_raises(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    """Without a CRS, degree coordinates would pass connect_tol (1 degree
    is ~100 km) and be written as if meters."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    # pyogrio itself warns about writing without a CRS; suppress that
    # unrelated warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _write_segments_shp(shp, synthetic_lines, [101, 102, 103], crs=None)
    with pytest.raises(ValueError, match="has no CRS"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )


@pytest.mark.domainless
def test_public_exports():
    import pywatershed as pws

    for name in (
        "at_a_station_hydraulic_geometry",
        "export_network_hydraulics",
        "shear_velocity",
        "calculate_seg_mid_elevations",
    ):
        assert callable(getattr(pws.utils, name))
        assert name in pws.utils.__all__


@pytest.mark.domainless
@pytest.mark.parametrize(
    "tosegment,match",
    [
        ([2, 1, 0], "Cycle"),  # seg0 -> seg1 -> seg0
        ([3, 3, -1], "tosegment"),  # negative
        ([3, 4, 0], "tosegment"),  # beyond nsegment
    ],
)
def test_calculate_seg_mid_elevations_bad_tosegment_raises(
    synthetic_params, tosegment, match
):
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["tosegment"] = np.array(tosegment, dtype=np.int64)
    bad = Parameters(**dd.data)
    with pytest.raises(ValueError, match=match):
        calculate_seg_mid_elevations(bad)


@pytest.mark.domainless
def test_export_out_of_range_tosegment_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["tosegment"] = np.array([3, 4, 0], dtype=np.int64)
    bad = Parameters(**dd.data)
    with pytest.raises(ValueError, match="tosegment"):
        export_network_hydraulics(bad, synthetic_run_dir, tmp_path / "net.nc")


@pytest.mark.domainless
def test_export_time_axis_mismatch_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    path = synthetic_run_dir / "seg_flow_depth.nc"
    with xr.open_dataarray(path) as opened:
        da = opened.load()  # close the file before rewriting it (Windows)
    da = da.assign_coords(time=TIMES + np.timedelta64(1, "D"))
    da.to_netcdf(path)
    with pytest.raises(ValueError, match="time"):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )


@pytest.mark.domainless
def test_export_to_id_masked_at_outlets(
    synthetic_params, synthetic_run_dir, tmp_path
):
    """A bogus tosegment_nhm at an outlet is forced to 0."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["tosegment_nhm"] = np.array([103, 103, 999], dtype=np.int64)
    params = Parameters(**dd.data)
    ds = xr.open_dataset(
        export_network_hydraulics(
            params, synthetic_run_dir, tmp_path / "net.nc"
        )
    )
    np.testing.assert_array_equal(ds["to_id"], np.array([103, 103, 0]))
    ds.close()


@pytest.mark.domainless
def test_export_to_id_disagreeing_with_tosegment_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    """A stale tosegment_nhm (e.g. after subsetting) would put two
    topologies in one file: to_id vs to_index."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    dd = synthetic_params.to_dd()
    # tosegment says reach 1 -> reach 2 (103); tosegment_nhm says 102
    dd.data_vars["tosegment_nhm"] = np.array([103, 102, 0], dtype=np.int64)
    params = Parameters(**dd.data)
    with pytest.raises(ValueError, match=r"indices \[1\].*gives \[102\]"):
        export_network_hydraulics(
            params, synthetic_run_dir, tmp_path / "net.nc"
        )


@pytest.mark.domainless
@pytest.mark.parametrize(
    "name,value,match",
    [
        ("seg_length", 0.0, "seg_length must be positive"),
        ("seg_width", np.nan, "seg_width must be positive"),
        ("mann_n", -0.03, "mann_n must be positive"),
        ("seg_depth", 0.0, "seg_depth must be positive"),
        ("seg_slope", -0.001, "seg_slope must be non-negative"),
    ],
)
def test_export_bad_static_parameter_raises(
    synthetic_params, synthetic_run_dir, tmp_path, name, value, match
):
    """Static parameters are written verbatim and consumers scale by
    length, so bad values must be rejected, naming the parameter."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    dd = synthetic_params.to_dd()
    values = dd.data_vars[name].astype(float).copy()
    values[1] = value
    dd.data_vars[name] = values
    with pytest.raises(ValueError, match=f"{match}.*\\[1\\]"):
        export_network_hydraulics(
            Parameters(**dd.data), synthetic_run_dir, tmp_path / "net.nc"
        )


@pytest.mark.domainless
def test_export_missing_nhm_seg_coord_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    path = synthetic_run_dir / "seg_outflow.nc"
    with xr.open_dataarray(path) as opened:
        da = opened.load()
    da = da.drop_vars("nhm_seg")
    da.to_netcdf(path)
    with pytest.raises(ValueError, match="nhm_seg"):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )


@pytest.mark.domainless
def test_shear_velocity_bad_slope_raises():
    from pywatershed.utils.network_hydraulics import shear_velocity

    with pytest.raises(ValueError, match="non-negative and finite"):
        shear_velocity(np.array([1.0, 2.0]), np.array([0.001, -0.01]))
    with pytest.raises(ValueError, match="non-negative and finite"):
        shear_velocity(np.array([1.0, 2.0]), np.array([0.001, np.nan]))
    with pytest.raises(ValueError, match="depth must be non-negative"):
        shear_velocity(np.array([1.0, -2.0]), np.array([0.001, 0.01]))


@pytest.mark.domainless
def test_export_units_mismatch_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    path = synthetic_run_dir / "seg_flow_depth.nc"
    with xr.open_dataarray(path) as opened:
        da = opened.load()
    da.attrs["units"] = "feet"
    da.to_netcdf(path)
    with pytest.raises(ValueError, match="units"):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )


@pytest.mark.domainless
def test_export_missing_units_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    """A file without units was not written by pywatershed; assuming
    units could scale already-SI data by CFS_TO_CMS a second time."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    path = synthetic_run_dir / "seg_outflow.nc"
    with xr.open_dataarray(path) as opened:
        da = opened.load()
    del da.attrs["units"]
    da.to_netcdf(path)
    with pytest.raises(ValueError, match="no units"):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )


def _poison_run_var(run_dir, name, value, itime=1, iseg=1):
    path = run_dir / f"{name}.nc"
    with xr.open_dataarray(path) as opened:
        da = opened.load()
    da.values[itime, iseg] = value
    da.to_netcdf(path)


@pytest.mark.domainless
@pytest.mark.parametrize(
    "name,value,match",
    [
        ("seg_flow_depth", np.nan, "non-finite or negative"),
        ("seg_outflow", -1.0, "non-finite or negative"),
        ("seg_res_time", np.inf, "non-finite or negative"),
        ("seg_tave_water", np.nan, "non-finite value"),
    ],
)
def test_export_bad_run_values_raise(
    synthetic_params, synthetic_run_dir, tmp_path, name, value, match
):
    """Fill values (a run that stopped early) or negatives must not reach
    the file or sqrt in shear_velocity."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    if name == "seg_tave_water":
        _write_run_var(
            synthetic_run_dir,
            name,
            np.full((NTIME, NSEG), 12.5),
            "degrees Celsius",
            synthetic_params.parameters["nhm_seg"],
        )
    _poison_run_var(synthetic_run_dir, name, value)
    with pytest.raises(ValueError, match=f"{name}.nc has 1 {match}"):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )


@pytest.mark.domainless
def test_export_negative_water_temperature_allowed(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    _write_run_var(
        synthetic_run_dir,
        "seg_tave_water",
        np.full((NTIME, NSEG), -0.5),
        "degrees Celsius",
        synthetic_params.parameters["nhm_seg"],
    )
    export_network_hydraulics(
        synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
    )
    with xr.open_dataset(tmp_path / "net.nc") as ds:
        assert float(ds["water_temperature"].min()) == -0.5


@pytest.mark.domainless
@pytest.mark.parametrize(
    "kwargs,match",
    [
        # the run is 1979-01-01..04; a window reaching outside it must
        # raise, not clip
        ({"start_time": np.datetime64("1978-12-31")}, "before the run"),
        ({"end_time": np.datetime64("1979-01-05")}, "after the run"),
        (
            {
                "start_time": np.datetime64("1979-01-01"),
                "end_time": np.datetime64("1979-01-05"),
            },
            "after the run",
        ),
        # inside the span but between daily steps: nothing selected
        (
            {
                "start_time": np.datetime64("1979-01-01T06"),
                "end_time": np.datetime64("1979-01-01T18"),
            },
            "no time steps between",
        ),
    ],
)
def test_export_time_window_outside_run_raises(
    synthetic_params, synthetic_run_dir, tmp_path, kwargs, match
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    with pytest.raises(ValueError, match=match):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc", **kwargs
        )


@pytest.mark.domainless
def test_export_inverted_time_bounds_raise(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    with pytest.raises(ValueError, match="after"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            start_time=np.datetime64("1979-01-03"),
            end_time=np.datetime64("1979-01-02"),
        )


@pytest.mark.domainless
def test_export_missing_required_parameters_raise(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    dd = synthetic_params.to_dd()
    for name in ("seg_width", "mann_n"):
        del dd.data_vars[name]
        del dd.metadata[name]
    short = Parameters(**dd.data)
    with pytest.raises(ValueError) as excinfo:
        export_network_hydraulics(
            short, synthetic_run_dir, tmp_path / "net.nc"
        )
    assert "seg_width" in str(excinfo.value)
    assert "mann_n" in str(excinfo.value)


@pytest.mark.domainless
def test_export_zero_flow_notes_and_connect_tol_attrs(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
    )
    assert ds["velocity"].attrs["note"] == (
        "0 where flow_out == 0; velocity is also 0 where width*depth <= "
        "1e-6 m2; mask on flow_out > 0"
    )
    for name in ("depth", "width", "residence_time"):
        assert "flow_out > 0" in ds[name].attrs["note"]
    assert ds.attrs["connect_tol"] == 1.0
    assert "mask on flow_out > 0" in ds.attrs["conventions_note"]
    ds.close()


@pytest.mark.domainless
def test_export_zero_flow_row(synthetic_params, synthetic_run_dir, tmp_path):
    """Zero flow and depth (a dry reach) must come through as 0, never as
    inf or NaN from a division or sqrt added to the export later."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    for name in ("seg_outflow", "seg_flow_depth"):
        path = synthetic_run_dir / f"{name}.nc"
        with xr.open_dataarray(path) as opened:
            da = opened.load()
        da.values[0, 0] = 0.0
        da.to_netcdf(path)
    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
    )
    assert ds["flow_out"].values[0, 0] == 0.0
    assert ds["ustar"].values[0, 0] == 0.0
    for name in ("flow_out", "velocity", "depth", "width", "ustar"):
        assert np.isfinite(ds[name].values).all()
    ds.close()


@pytest.mark.domainless
def test_export_polyline_multilinestring_raises(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from shapely.geometry import MultiLineString

    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    gdf = gpd.GeoDataFrame(
        {"nsegment_v": [101, 102, 103], "model_idx": [1, 2, 3]},
        geometry=[
            MultiLineString(
                [[(-1000.0, 0.0), (-600.0, 0.0)], [(-400.0, 0.0), (0.0, 0.0)]]
            ),
            LineString(synthetic_lines[1]),
            LineString(synthetic_lines[2]),
        ],
        crs="EPSG:5070",
    )
    gdf.to_file(shp)
    with pytest.raises(ValueError, match="LineString"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )


@pytest.mark.domainless
def test_export_polyline_mostly_unconnected_raises(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    lines = [list(ll) for ll in synthetic_lines]
    lines[0] = [(-1000.0, 50.0), (0.0, 50.0)]
    lines[1] = [(50.0, 0.0), (550.0, 1000.0), (1050.0, 2000.0)]
    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, lines, [101, 102, 103])
    with pytest.raises(ValueError, match="mismatch"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )


@pytest.mark.domainless
def test_export_polyline_outlet_without_upstream_warns(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    dd = synthetic_params.to_dd()
    # reach 1 becomes an outlet with no upstream reach
    dd.data_vars["tosegment"] = np.array([3, 0, 0], dtype=np.int64)
    params = Parameters(**dd.data)
    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, synthetic_lines, [101, 102, 103])
    with pytest.warns(UserWarning, match="no upstream"):
        out = export_network_hydraulics(
            params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )
    ds = xr.open_dataset(out)
    assert ds.attrs["n_unconnected"] == 0
    ds.close()


@pytest.mark.domainless
def test_export_polyline_connect_tol_boundary(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    """A gap of exactly connect_tol is connected: the test is strict >."""
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    lines = [list(ll) for ll in synthetic_lines]
    lines[0] = [(-1000.0, 1.0), (0.0, 1.0)]  # displaced by exactly 1 m
    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, lines, [101, 102, 103])
    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )
    )
    assert ds.attrs["n_unconnected"] == 0
    ds.close()


@pytest.mark.domainless
def test_export_polyline_duplicate_ids_raise(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, synthetic_lines, [101, 101, 103])
    with pytest.raises(ValueError, match="nsegment_v"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )


@pytest.mark.domainless
def test_mmr_to_mf6_dfw_seg_mid_elevation_check(synthetic_params):
    """The debug check path runs and sets both attributes."""
    from pywatershed.utils.mmr_to_mf6_dfw import MmrToMf6Dfw

    fake = types.SimpleNamespace(parameters=synthetic_params)
    MmrToMf6Dfw._calculate_seg_mid_elevations(fake, check=True)
    np.testing.assert_allclose(
        fake._seg_mid_elevation, np.array([108.0, 108.0, 101.5])
    )
    assert fake._outlet_chds == {2: 102.5}


@pytest.mark.domainless
@pytest.mark.parametrize("name", ["seg_slope", "seg_length", "hru_elev"])
def test_calculate_seg_mid_elevations_non_finite_raises(
    synthetic_params, name
):
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    dd = synthetic_params.to_dd()
    values = dd.data_vars[name].astype(float).copy()
    values[1] = np.nan
    dd.data_vars[name] = values
    with pytest.raises(ValueError, match=f"{name} has non-finite"):
        calculate_seg_mid_elevations(Parameters(**dd.data))


@pytest.mark.domainless
@pytest.mark.parametrize(
    "bad_mid,match",
    [
        ([110.0, 108.0, 101.5], "Segment 0"),  # interior rise wrong
        ([110.0, 110.0, 103.5], "Outlet segment 2"),  # outlet too high
        ([np.nan, 108.0, 101.5], "Segment 0"),  # NaN must not pass
        # a NaN outlet fails the upstream segment's check first
        ([108.0, 108.0, np.nan], "Segment 0"),
    ],
)
def test_mmr_to_mf6_dfw_seg_mid_elevation_check_raises(
    synthetic_params, monkeypatch, bad_mid, match
):
    """The debug check raises ValueError on a broken invariant."""
    from pywatershed.utils import mmr_to_mf6_dfw as mod

    mid = np.array(bad_mid)
    monkeypatch.setattr(
        mod,
        "calculate_seg_mid_elevations",
        lambda parameters: (mid, {2: float(mid[2])}),
    )
    fake = types.SimpleNamespace(parameters=synthetic_params)
    with pytest.raises(ValueError, match=match):
        mod.MmrToMf6Dfw._calculate_seg_mid_elevations(fake, check=True)
