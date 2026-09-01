# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the cad2roi package (pure numpy/cv2/PIL — no models).

Covers the CAD mask parser, mask/morphology utilities, and the per-defect ROI-candidate factories
with exact-value assertions. The AMP-invoking placers (LessSolder/Bridge ``place_on_roi``) drive the
separately-tested placement engine with retries, so only their deterministic helpers are checked here.
"""

import json
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from anomalygen.auto_mask_placement.cad2roi import (
    CADComponent,
    CADParser,
    CADToROIGenerator,
    ExcessSolderMaskPlacer,
    LessSolderMaskPlacer,
    MissingMaskPlacer,
    ROICandidate,
    get_bridge_candidates,
    get_bridge_groups,
    get_excess_solder_candidates,
    get_less_solder_candidates,
    get_missing_candidates,
    visualize_roi_candidates,
)
from anomalygen.auto_mask_placement.cad2roi.defects import bridge
from anomalygen.auto_mask_placement.cad2roi.defects.bridge import (
    _check_single_component,
    _check_touches_regions,
)
from anomalygen.auto_mask_placement.cad2roi.defects.less_solder import (
    _check_min_area,
)
from anomalygen.auto_mask_placement.cad2roi.defects.less_solder import (
    _validate_placement as _ls_validate_placement,
)
from anomalygen.auto_mask_placement.cad2roi.mask_utils import (
    load_and_scale_submask,
    mask_area,
    mask_bbox,
    merge_class_masks,
    read_amp_output,
    save_temp_mask,
)
from anomalygen.auto_mask_placement.cad2roi.morph_ops import close, dilate, erode
from anomalygen.auto_mask_placement.cad2roi.visualize import _blend_overlay

# --- helpers -------------------------------------------------------------------------------------


def _rect_mask(shape, boxes):
    """uint8 0/255 mask with a filled rect per (x, y, w, h) in ``boxes``."""
    m = np.zeros(shape, np.uint8)
    for x, y, w, h in boxes:
        m[y : y + h, x : x + w] = 255
    return m


def _component(class_name, mask, comp_id=0):
    x, y, w, h = mask_bbox(mask)
    ys, xs = np.nonzero(mask)
    return CADComponent(
        class_name=class_name,
        mask=mask,
        bbox=(x, y, w, h),
        area=mask_area(mask),
        centroid=(float(xs.mean()), float(ys.mean())),
        component_id=comp_id,
    )


# --- mask_utils ----------------------------------------------------------------------------------


def test_mask_area_counts_foreground():
    assert mask_area(_rect_mask((50, 50), [(10, 10, 5, 4)])) == 20  # 5*4
    assert mask_area(np.zeros((10, 10), np.uint8)) == 0


def test_mask_bbox_tight_and_empty():
    assert mask_bbox(_rect_mask((50, 60), [(7, 9, 5, 8)])) == (7, 9, 5, 8)
    assert mask_bbox(np.zeros((10, 10), np.uint8)) is None


def test_merge_class_masks_unions_selected_classes():
    a = _rect_mask((40, 40), [(2, 2, 6, 6)])
    b = _rect_mask((40, 40), [(20, 20, 5, 5)])
    c = _rect_mask((40, 40), [(30, 5, 4, 4)])
    components = {"pad": [_component("pad", a)], "solder": [_component("solder", b)], "ic": [_component("ic", c)]}
    merged = merge_class_masks(components, ("pad", "solder"), (40, 40))
    assert np.array_equal(merged, cv2.bitwise_or(a, b))  # ic excluded
    assert mask_area(merged) == mask_area(a) + mask_area(b)  # disjoint


def test_save_temp_mask_roundtrips():
    mask = _rect_mask((30, 30), [(5, 5, 10, 8)])
    path = save_temp_mask(mask)
    try:
        assert np.array_equal(cv2.imread(path, cv2.IMREAD_GRAYSCALE), mask)
    finally:
        os.unlink(path)


def test_load_and_scale_submask_scales_to_target_area_and_centers():
    submask = save_temp_mask(_rect_mask((100, 100), [(40, 40, 20, 20)]))  # area 400
    try:
        # scale = sqrt(target/area) = sqrt(1600/400) = 2 -> 40x40 solid = area 1600, centered.
        out_path = load_and_scale_submask(submask, target_area=1600, img_shape=(100, 100), scale_factor=1.0)
        out = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        os.unlink(out_path)
    finally:
        os.unlink(submask)
    assert mask_area(out) == 1600
    assert mask_bbox(out) == (30, 30, 40, 40)  # centered on (50, 50)


def test_read_amp_output(tmp_path):
    # Empty dir -> None.
    assert read_amp_output(str(tmp_path)) is None

    # Non-empty mask -> returned, and the files are cleaned up.
    mask = _rect_mask((20, 20), [(2, 2, 5, 5)])
    cv2.imwrite(str(tmp_path / "out.png"), mask)
    got = read_amp_output(str(tmp_path))
    assert got is not None and np.array_equal(got, mask)
    assert list(tmp_path.glob("*.png")) == []  # consumed

    # All-zero mask -> None.
    cv2.imwrite(str(tmp_path / "zero.png"), np.zeros((20, 20), np.uint8))
    assert read_amp_output(str(tmp_path)) is None


# --- morph_ops -----------------------------------------------------------------------------------


def test_dilate_grows_and_erode_shrinks():
    mask = _rect_mask((60, 60), [(20, 20, 20, 20)])
    assert mask_area(dilate(mask, 5)) > mask_area(mask)
    assert mask_area(erode(mask, 5)) < mask_area(mask)


def test_close_bridges_small_gap():
    # Two blobs 2px apart -> two components; closing joins them into one.
    two = _rect_mask((40, 60), [(10, 15, 10, 10), (22, 15, 10, 10)])
    assert cv2.connectedComponentsWithStats(two)[0] - 1 == 2
    joined = close(two, 7)
    assert cv2.connectedComponentsWithStats(joined)[0] - 1 == 1


# --- CADParser -----------------------------------------------------------------------------------


def _write_label(tmp_path):
    label = {
        "(0, 0, 0, 255)": {"class": "BACKGROUND"},
        "(255, 0, 0, 255)": {"class": "pad"},
        "(0, 255, 0, 255)": {"class": "solder"},
    }
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(label))
    return str(p)


def _write_cad_mask(tmp_path, colored_boxes, shape=(60, 80)):
    """RGB PNG (H, W); colored_boxes: list of ((x, y, w, h), (r, g, b))."""
    img = np.zeros((shape[0], shape[1], 3), np.uint8)
    for (x, y, w, h), color in colored_boxes:
        img[y : y + h, x : x + w] = color
    p = tmp_path / "mask.png"
    Image.fromarray(img).save(p)
    return str(p)


def test_cad_parser_classifies_and_extracts_components(tmp_path):
    label = _write_label(tmp_path)
    mask = _write_cad_mask(
        tmp_path,
        [((10, 10, 20, 20), (255, 0, 0)), ((40, 10, 20, 20), (0, 255, 0))],  # red pad, green solder
    )
    parser = CADParser(label)
    assert parser.class_names == ["pad", "solder"]  # BACKGROUND dropped

    components, shape = parser.parse(mask)
    assert shape == (60, 80)
    assert set(components) == {"pad", "solder"}
    assert len(components["pad"]) == 1 and len(components["solder"]) == 1

    pad = components["pad"][0]
    assert pad.class_name == "pad"
    assert pad.bbox == (10, 10, 20, 20)
    assert pad.area == 400
    assert pad.centroid[0] == pytest.approx(19.5, abs=0.5)
    assert components["solder"][0].bbox == (40, 10, 20, 20)


def test_cad_parser_drops_fragments(tmp_path):
    label = _write_label(tmp_path)
    # One big red pad (400) and one tiny red fragment (16 < 400*0.1) -> fragment dropped.
    mask = _write_cad_mask(
        tmp_path,
        [((10, 10, 20, 20), (255, 0, 0)), ((60, 40, 4, 4), (255, 0, 0))],
    )
    parser = CADParser(label, min_area_abs=5, fragment_ratio=0.1)
    components, _ = parser.parse(mask)
    assert len(components["pad"]) == 1
    assert components["pad"][0].area == 400


# --- defect factories ----------------------------------------------------------------------------


def test_get_less_solder_candidates_one_per_pad():
    pads = [
        _component("pad", _rect_mask((40, 40), [(2, 2, 6, 6)]), 0),
        _component("pad", _rect_mask((40, 40), [(20, 20, 5, 5)]), 1),
    ]
    components = {"pad": pads}
    cands = get_less_solder_candidates(components, (40, 40))
    assert len(cands) == 2
    assert [c.component_ids for c in cands] == [[0], [1]]
    assert cands[0].source_classes == ["pad"]
    assert np.array_equal(cands[0].mask, pads[0].mask)
    assert cands[1].area == pads[1].area


def test_get_bridge_groups_merges_connected_pad_solder():
    # Two separate pad+solder clusters -> two groups.
    pad1 = _rect_mask((60, 100), [(10, 20, 10, 10)])
    sol1 = _rect_mask((60, 100), [(20, 20, 10, 10)])  # touches pad1
    pad2 = _rect_mask((60, 100), [(70, 20, 10, 10)])
    components = {
        "pad": [_component("pad", pad1, 0), _component("pad", pad2, 1)],
        "solder": [_component("solder", sol1, 0)],
    }
    groups = get_bridge_groups(components, (60, 100), classes=("pad", "solder"))
    assert len(groups) == 2
    areas = sorted(g.area for g in groups)
    assert areas == [100, 200]  # pad2 alone (100), pad1+sol1 merged (200)


def test_get_excess_solder_delegates_to_groups():
    pad = _rect_mask((40, 60), [(10, 10, 10, 10)])
    sol = _rect_mask((40, 60), [(20, 10, 10, 10)])  # touches pad -> one group
    components = {"pad": [_component("pad", pad, 0)], "solder": [_component("solder", sol, 0)]}
    cands = get_excess_solder_candidates(components, (40, 60))
    assert len(cands) == 1
    assert np.array_equal(cands[0].mask, cv2.bitwise_or(pad, sol))


def test_excess_solder_placer_is_less_solder_placer():
    assert ExcessSolderMaskPlacer is LessSolderMaskPlacer


def test_get_bridge_candidates_needs_two_groups():
    # Only one pad -> a single group -> no bridge possible.
    components = {"pad": [_component("pad", _rect_mask((40, 40), [(10, 10, 10, 10)]), 0)]}
    assert get_bridge_candidates(components, (40, 40)) == []


def test_get_missing_candidates_requires_component_in_unit():
    shape = (60, 100)
    cap = _rect_mask(shape, [(10, 20, 20, 20)])
    sol_near = _rect_mask(shape, [(30, 20, 12, 20)])  # touches capacitor -> same unit
    sol_far = _rect_mask(shape, [(80, 45, 10, 10)])  # lone solder, no capacitor/ic
    components = {
        "capacitor": [_component("capacitor", cap, 0)],
        "solder": [_component("solder", sol_near, 0), _component("solder", sol_far, 1)],
    }
    cands = get_missing_candidates(components, shape)
    # Exactly one unit contains a capacitor; the lone-solder unit is rejected.
    assert len(cands) == 1
    # The candidate covers the capacitor region.
    assert cv2.bitwise_and(cands[0].mask, cap).sum() > 0
    assert cv2.bitwise_and(cands[0].mask, sol_far).sum() == 0


# --- MissingMaskPlacer (deterministic; no AMP) ---------------------------------------------------


def _disjoint_candidates(n=3):
    cands = []
    for i in range(n):
        m = _rect_mask((60, 60), [(2 + i * 15, 2, 10, 10)])
        cands.append(ROICandidate(mask=m, bbox=mask_bbox(m), centroid=(0, 0), area=mask_area(m), component_ids=[i]))
    return cands


def test_missing_placer_empty_returns_empty():
    assert MissingMaskPlacer().place_all([]) == {}


def test_missing_placer_selects_and_ors_masks_deterministically():
    cands = _disjoint_candidates(3)
    result = MissingMaskPlacer().place_all(cands, n_instances=2, n_seeds=1, base_seed=42)

    # Replicate the exact selection the placer makes for seed 42.
    expected_idx = np.random.RandomState(42).choice(3, size=2, replace=False)
    expected = np.zeros((60, 60), np.uint8)
    for i in expected_idx:
        expected = cv2.bitwise_or(expected, cands[i].mask)

    assert set(result) == {1}
    n_sel, n_placed, combined = result[1]
    assert (n_sel, n_placed) == (2, 2)
    assert np.array_equal(combined, expected)


# --- pure placement helpers ----------------------------------------------------------------------


def test_check_min_area():
    roi = _rect_mask((40, 40), [(0, 0, 10, 10)])  # area 100
    assert _check_min_area(_rect_mask((40, 40), [(0, 0, 10, 6)]), mask_area(roi), 0.5) is True  # 60 >= 50
    assert _check_min_area(_rect_mask((40, 40), [(0, 0, 10, 4)]), mask_area(roi), 0.5) is False  # 40 < 50
    assert _check_min_area(roi, 0, 0.5) is True  # roi_area <= 0 short-circuits


def test_less_solder_validate_placement():
    roi = _rect_mask((40, 40), [(0, 0, 10, 10)])  # area 100
    roi_area = mask_area(roi)
    # Inside ROI and above the min-area ratio -> returns the clip (placed & roi).
    placed = _rect_mask((40, 40), [(0, 0, 10, 8)])  # area 80 >= 50
    clipped = _ls_validate_placement(placed, roi, roi_area, 0.5)
    assert clipped is not None and np.array_equal(clipped, cv2.bitwise_and(placed, roi))
    # Clipped area below the ratio -> None.
    assert _ls_validate_placement(_rect_mask((40, 40), [(0, 0, 10, 3)]), roi, roi_area, 0.5) is None  # area 30
    # No overlap with the ROI -> None.
    assert _ls_validate_placement(_rect_mask((40, 40), [(20, 20, 5, 5)]), roi, roi_area, 0.5) is None


def test_bridge_check_touches_and_single_component():
    r1 = _rect_mask((40, 60), [(2, 2, 8, 8)])
    r2 = _rect_mask((40, 60), [(40, 2, 8, 8)])
    touching_both = cv2.bitwise_or(_rect_mask((40, 60), [(2, 4, 8, 2)]), _rect_mask((40, 60), [(40, 4, 8, 2)]))
    assert _check_touches_regions(touching_both, [r1, r2], min_count=2) is True
    assert _check_touches_regions(_rect_mask((40, 60), [(2, 4, 8, 2)]), [r1, r2], min_count=2) is False  # only r1

    assert _check_single_component(_rect_mask((40, 40), [(5, 5, 10, 10)])) is True
    assert _check_single_component(_rect_mask((40, 40), [(2, 2, 5, 5), (30, 30, 5, 5)])) is False


# --- visualize -----------------------------------------------------------------------------------


def test_blend_overlay_math():
    img = np.zeros((10, 10, 3), np.uint8)
    mask = _rect_mask((10, 10), [(0, 0, 5, 5)])
    out = _blend_overlay(img, mask, color=(100, 0, 0), alpha=0.6)
    # Inside mask: 0*0.4 + 100*0.6 = 60. Outside: unchanged.
    assert out[0, 0, 0] == 60
    assert out[9, 9, 0] == 0


def test_visualize_roi_candidates_writes_expected_files(tmp_path):
    cad = _write_cad_mask(tmp_path, [((10, 10, 20, 20), (255, 0, 0))], shape=(60, 60))
    m = _rect_mask((60, 60), [(10, 10, 20, 20)])
    cand = ROICandidate(mask=m, bbox=mask_bbox(m), centroid=(20, 20), area=mask_area(m), component_ids=[0])
    out_dir = tmp_path / "viz"
    visualize_roi_candidates(cad, {"missing": [cand], "less_solder": [], "bridge": []}, str(out_dir), mask_name="cad0")

    assert (out_dir / "cad0_missing_overview.png").exists()
    assert (out_dir / "cad0_missing_roi_0.png").exists()
    assert (out_dir / "cad0_summary.png").exists()
    # The per-ROI mask is written verbatim.
    assert np.array_equal(cv2.imread(str(out_dir / "cad0_missing_roi_0.png"), cv2.IMREAD_GRAYSCALE), m)


# --- CADToROIGenerator (end-to-end through the real parser) --------------------------------------


def test_generator_produces_all_defect_types(tmp_path):
    label = {
        "(0, 0, 0, 255)": {"class": "BACKGROUND"},
        "(255, 0, 0, 255)": {"class": "pad"},
        "(0, 255, 0, 255)": {"class": "solder"},
        "(0, 0, 255, 255)": {"class": "capacitor"},
    }
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps(label))

    mask = _write_cad_mask(
        tmp_path,
        [
            ((10, 10, 20, 20), (0, 0, 255)),  # capacitor  ]-> missing unit
            ((30, 10, 12, 20), (0, 255, 0)),  # solder touching the capacitor
            ((60, 10, 15, 15), (255, 0, 0)),  # pad 1      ]-> less_solder
            ((95, 10, 15, 15), (255, 0, 0)),  # pad 2
        ],
        shape=(80, 120),
    )

    result = CADToROIGenerator(str(label_path)).generate_all_candidates(mask)

    assert set(result) == {"missing", "less_solder", "excess_solder", "bridge"}
    assert all(isinstance(v, list) for v in result.values())
    assert len(result["missing"]) >= 1  # capacitor + adjacent solder
    assert len(result["less_solder"]) == 2  # two pad components


def test_cadparser_parses_valid_color_keys_and_rejects_non_literal(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "(255, 0, 0, 255)": {"class": "pad"},
                "255,0,0,255": {"class": "solder"},
                "(0, 0, 0, 255)": {"class": "BACKGROUND"},
            }
        )
    )
    parser = CADParser(str(valid))
    assert set(parser.class_names) == {"pad", "solder"}
    np.testing.assert_array_equal(parser.color_map["pad"], np.array([255, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(parser.color_map["solder"], np.array([255, 0, 0], dtype=np.float32))

    # A non-literal key must be rejected, not executed (the old eval() would run it).
    malicious = tmp_path / "malicious.json"
    malicious.write_text(json.dumps({"__import__('os').system('echo pwned')": {"class": "pad"}}))
    with pytest.raises(ValueError):
        CADParser(str(malicious))

    # A valid-but-wrong-type literal (set / scalar) parses fine but is not an (r, g, b) sequence ->
    # must raise a clean ValueError here, not a cryptic TypeError later at rgba[:3].
    for bad_key in ("{1, 2, 3}", "5"):
        wrong_type = tmp_path / "wrong_type.json"
        wrong_type.write_text(json.dumps({bad_key: {"class": "pad"}}))
        with pytest.raises(ValueError):
            CADParser(str(wrong_type))


def test_bridge_chains_groups_along_principal_axis(monkeypatch):
    h = w = 16

    def _grp(cx, cy):
        m = np.zeros((h, w), dtype=np.uint8)
        yi, xi = int(round(cy)), int(round(cx))
        m[yi : yi + 2, xi : xi + 2] = 255
        return ROICandidate(mask=m, bbox=(xi, yi, 2, 2), centroid=(float(cx), float(cy)), area=4)

    # Vertical column with x-jitter: y = 10, 0, 5 at indices 0, 1, 2.
    groups = [_grp(0, 10), _grp(1, 0), _grp(0, 5)]

    chain = []
    monkeypatch.setattr(
        bridge,
        "_fill_gap_between",
        lambda merged, ga, gb, shape: chain.append((ga.centroid, gb.centroid)) or merged,
    )
    bridge._try_bridge_roi(
        [0, 1, 2],
        groups,
        np.zeros((h, w), dtype=np.uint8),
        np.zeros((h, w), dtype=np.uint8),
        (h, w),
        fill_gap=True,
        bridge_max_cut_ratio=0.5,
    )

    # Reconstruct the visited y-order from the consecutive gap-fill pairs.
    assert len(chain) == 2
    ys = [chain[0][0][1]] + [b[1] for _, b in chain]
    assert ys == sorted(ys) or ys == sorted(ys, reverse=True), f"chain not monotonic: {ys}"
