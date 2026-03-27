from project.core.recognition.roi.geometry import check_observation_plausibility
from project.core.recognition.roi.observation import RoiObservation
from project.data.data_models import Ellipse2D, Point2D


def _make_obs(
    *,
    pupil_center_xy,
    iris_center_xy,
    pupil_axes,
    iris_axes,
    pupil_inside_iris_ratio,
    pupil_center_inside_iris_d2,
):
    pupil_el = Ellipse2D(
        cx=float(pupil_center_xy[0]),
        cy=float(pupil_center_xy[1]),
        major_axis=float(pupil_axes[0]),
        minor_axis=float(pupil_axes[1]),
        angle_deg=0.0,
        confidence=1.0,
    )
    iris_el = Ellipse2D(
        cx=float(iris_center_xy[0]),
        cy=float(iris_center_xy[1]),
        major_axis=float(iris_axes[0]),
        minor_axis=float(iris_axes[1]),
        angle_deg=0.0,
        confidence=1.0,
    )
    pupil_center = Point2D(x=float(pupil_center_xy[0]), y=float(pupil_center_xy[1]))
    iris_center = Point2D(x=float(iris_center_xy[0]), y=float(iris_center_xy[1]))
    return RoiObservation(
        side="left",
        roi_xywh=(0, 0, 44, 19),
        roi_shape_hw=(19, 44),
        pupil_center_img=pupil_center,
        iris_center_img=iris_center,
        pupil_ellipse_roi=pupil_el,
        iris_ellipse_roi=iris_el,
        pupil_ellipse_img=pupil_el,
        iris_ellipse_img=iris_el,
        pupil_mask_bbox=None,
        iris_mask_bbox=None,
        union_mask_bbox=None,
        pupil_mask_centroid=None,
        iris_mask_centroid=None,
        union_mask_centroid=None,
        pupil_overlap_score=0.0,
        iris_overlap_score=0.0,
        pupil_inside_iris_ratio=float(pupil_inside_iris_ratio),
        pupil_center_inside_iris_score=max(0.0, 1.0 - float(pupil_center_inside_iris_d2)),
        pupil_center_inside_iris_d2=float(pupil_center_inside_iris_d2),
    )


def test_plausibility_accepts_normal_geometry():
    obs = _make_obs(
        pupil_center_xy=(17.65, 14.12),
        iris_center_xy=(17.89, 13.92),
        pupil_axes=(5.99, 3.63),
        iris_axes=(17.21, 10.90),
        pupil_inside_iris_ratio=0.95,
        pupil_center_inside_iris_d2=0.02,
    )
    ok, reason = check_observation_plausibility(obs)
    assert ok is True
    assert reason == ""


def test_plausibility_rejects_large_center_offset():
    obs = _make_obs(
        pupil_center_xy=(8.49, 13.91),
        iris_center_xy=(30.21, 14.81),
        pupil_axes=(6.36, 3.89),
        iris_axes=(15.17, 9.28),
        pupil_inside_iris_ratio=0.10,
        pupil_center_inside_iris_d2=8.25,
    )
    ok, reason = check_observation_plausibility(obs)
    assert ok is False
    assert reason in {"pupil_boundary_outside_iris", "pupil_center_outside_iris", "pupil_iris_center_offset_too_large"}
