from project.data.data_manager import CameraDataManager
from project.data.data_models import Landmark, EYE_TYPE, FITTING_TYPE, BGRImage
from typing import Dict, Any
from project.core.recognition.stereo_recognizer import StereoRecognizer
from time import sleep
import cv2
import json
import os
from project.managers import FRAME_ID_MANAGER, FrameIdManager, DATA_PIPELINE_MANAGER, CALLBACK_MANAGER
from project.data.data_manager import RecgFitDataManager
from project.events import RECOGNITION_COMPLETE

# runtime visualization helpers (pure UI/drawing)
from project.core.recognition.runtime_visualization import (
    scale_landmarks_for_display,
    draw_bbox_and_rois,
    ellseg_segmentation_mask_to_bgr,
    native_geometry_for_roi_crop,
    draw_ellseg_native_geometry,
    draw_roi_debug_overlay,
    show_frame,
)
# 配置日志
from project.config.logging_config import setup_logging
import logging
logger = setup_logging(__name__, logging.DEBUG)



class RecognitionException(Exception):
    """识别异常"""
    def __init__(self, message, error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class RecognitionWarning(RecognitionException):
    """警告级别异常 - 可恢复的问题"""
    def __init__(self, message, error_code=None, retry_count=0, sleep_duration=None):
        super().__init__(message, error_code)
        self.retry_count = retry_count
        self.severity = "WARNING"
        self.sleep_duration = sleep_duration  # 可选的睡眠时间

class RecognitionError(RecognitionException):
    """错误级别异常 - 严重问题，需要停止处理"""
    def __init__(self, message, error_code=None, should_stop=True):
        super().__init__(message, error_code)
        self.severity = "ERROR"
        self.should_stop = should_stop  # 是否应该停止执行

class ErrorHandler:
    """错误处理策略 - 通过回调函数处理具体行为"""
    def __init__(self, max_warnings=10, max_retries=3):
        self.max_warnings = max_warnings
        self.max_retries = max_retries
        self.warning_count = 0
        self.retry_count = 0
        
        # 回调函数 - 由外部设置具体行为
        self.on_warning = None  # 警告时的睡眠回调
        self.on_error = None     # 错误时的停止回调
    
    def set_callbacks(self, on_warning=None, on_error=None):
        """设置回调函数"""
        self.on_warning = on_warning
        self.on_error = on_error
    
    def handle_warning(self, warning: RecognitionWarning) -> bool:
        """处理警告，返回是否继续执行"""
        self.warning_count += 1
        logger.warning(f"警告 {self.warning_count}/{self.max_warnings}: {warning.message}")
        
        if self.warning_count >= self.max_warnings:
            logger.error(f"警告次数超过限制 ({self.max_warnings})，停止执行")
            return False
        
        # 执行警告回调（如睡眠）
        if self.on_warning and warning.sleep_duration:
            self.on_warning(warning.sleep_duration)
        
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            logger.info(f"尝试重试 ({self.retry_count}/{self.max_retries})")
            return True
        
        return True
    
    def handle_error(self, error: RecognitionError) -> bool:
        """处理错误，返回是否继续执行"""
        logger.error(f"严重错误: {error.message}")
        
        # 执行错误回调（如停止运行）
        if self.on_error and error.should_stop:
            self.on_error()
        
        return False
    
    def reset_warnings(self):
        """重置警告计数"""
        self.warning_count = 0
        self.retry_count = 0

class RecognitionManager:
    def __init__(self, frame_id_manager: FrameIdManager = FRAME_ID_MANAGER, rgb_d=True):
        self.rgb_d = rgb_d
        self.camera_manager = CameraDataManager(rgb_d=rgb_d)
        self.stereo_recognizer = StereoRecognizer(self.camera_manager)
        self.running = False
        self.interval = 0.1
        self.error_handler = ErrorHandler()
        self.frame_id_manager = frame_id_manager
        self.data_pipeline = DATA_PIPELINE_MANAGER
        # 结构化 jsonl 文件句柄（延迟打开）
        self._roi_debug_files = {
            "left": None,
            "right": None,
        }
        # 设置回调函数
        self.error_handler.set_callbacks(
            on_warning=self._handle_warning,
            on_error=self._handle_error
        )
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """设置事件处理器"""
        from project.events import SYSTEM_STOP
        CALLBACK_MANAGER.register(SYSTEM_STOP, self._on_system_stop)
    
    def _cleanup_event_handlers(self):
        """清理事件处理器"""
        from project.events import SYSTEM_STOP
        CALLBACK_MANAGER.unregister(SYSTEM_STOP, self._on_system_stop)
    
    def _handle_warning(self, sleep_duration=None):
        """警告时的睡眠处理 - 接受 ErrorHandler 传入的 sleep_duration"""
        if sleep_duration is not None and sleep_duration > 0:
            sleep(sleep_duration)
    
    def _handle_error(self):
        """错误时的停止处理"""
        self.running = False
    
    def _check_camera_initialization(self):
        """检查摄像头初始化"""
        self.cap = self.camera_manager.get_cap()
        if self.cap is None or not self.cap.isOpened():
            # 如果摄像头对象无效，尝试重新初始化
            if not self.camera_manager.initialize_camera():
                raise RecognitionError("摄像头初始化失败", "CAM_INIT_FAILED")
    
    def _check_frame_reading(self, ret, frame):
        """检查帧读取"""
        if not ret:
            raise RecognitionWarning("无法读取摄像头数据", "FRAME_READ_FAILED", sleep_duration=self.interval)
    
    def _check_data_availability(self, image):
        """检查数据可用性"""
        if not image:
            raise RecognitionWarning("无法获取图像", "DATA_UNAVAILABLE", sleep_duration=self.interval)

    def run(self, warmup_frames: int = 20, max_frames: int = None, recognition_debug: bool = False):
        """主运行循环。recognition_debug=True 时每帧将绘制后的识别图保存到当前 debug 目录。"""
        self.running = True
        
        try:
            self._check_camera_initialization()
            self.cap = self.camera_manager.get_cap()
            
        except RecognitionError as e:
            logger.error(f"摄像头初始化失败: {e}", exc_info=True)
            self.error_handler.handle_error(e)
            return
    

        _win_name_left = "Gaze-Camera-Left"  # 左目窗口（ASCII 标题避免 Windows 下中文乱码）
        _win_name_right = "Gaze-Camera-Right"  # 右目窗口
        cv2.namedWindow(_win_name_left, cv2.WINDOW_NORMAL)
        cv2.namedWindow(_win_name_right, cv2.WINDOW_NORMAL)
        _display_params_logged_left = [False]
        _display_params_logged_right = [False]
        frames_processed = 0
        while self.running:
            try:
                frame_id = self._cycle_start()
                ret, frame = self.cap.read()
                self._check_frame_reading(ret, frame)
                frames_processed += 1
                if frames_processed < warmup_frames:
                    continue
                if max_frames is not None and frames_processed >= max_frames:
                    break
                self.camera_manager.add_frame(frame_id, frame)

                pair = self.camera_manager.get_stereo_pair(frame_id)
                if pair is None:
                    raise RecognitionWarning(
                        "非双目帧或无法获取左右图",
                        "STEREOPAIR_UNAVAILABLE",
                        sleep_duration=self.interval,
                    )
                left_image, right_image = pair
                self._check_data_availability(left_image)
                self._check_data_availability(right_image)
                display_image_left = left_image
                display_image_right = right_image
                success, key_coordinates = self.stereo_recognizer.recognize(
                    left_image, right_image, frame_id=frame_id
                )
                debug_2d = self.stereo_recognizer.get_last_debug_2d()

                # 左图：画人脸 bbox + left_roi 和 left_points_dict
                frame_display_left, display_w_l, display_h_l, w_l, h_l = show_frame(
                    _win_name_left, display_image_left, _display_params_logged_left, show=False
                )
                # 从 stereo_recognizer 调试数据中取 ROI / face bbox
                left_roi = debug_2d.get("left", {}).get("roi")
                face_bbox_left = debug_2d.get("face_bbox_left")
                frame_display_left = draw_bbox_and_rois(
                    frame_display_left, w_l, h_l, display_w_l, display_h_l,
                    bbox_rois=(face_bbox_left, left_roi, None),
                )
                left_debug = debug_2d.get("left", {}).get("debug", {})
                left_debug_right = debug_2d.get("left", {}).get("debug_right_eye", {})
                frame_display_left = draw_roi_debug_overlay(
                    frame_display_left, left_debug, left_debug_right, w_l, h_l, display_w_l, display_h_l
                )
                show_frame(
                    _win_name_left,
                    display_image_left,
                    _display_params_logged_left,
                    frame_to_show=frame_display_left,
                )

                # 右图：画人脸 bbox + right_roi 和 right_points_dict，全部落在右图上
                frame_display_right, display_w_r, display_h_r, w_r, h_r = show_frame(
                    _win_name_right, display_image_right, _display_params_logged_right, show=False
                )
                right_roi = debug_2d.get("right", {}).get("roi")
                face_bbox_right = debug_2d.get("face_bbox_right")
                frame_display_right = draw_bbox_and_rois(
                    frame_display_right, w_r, h_r, display_w_r, display_h_r,
                    bbox_rois=(face_bbox_right, None, right_roi),
                )
                right_debug = debug_2d.get("right", {}).get("debug", {})
                right_debug_right = debug_2d.get("right", {}).get("debug_right_eye", {})
                frame_display_right = draw_roi_debug_overlay(
                    frame_display_right, right_debug_right, right_debug, w_r, h_r, display_w_r, display_h_r
                )
                show_frame(
                    _win_name_right,
                    display_image_right,
                    _display_params_logged_right,
                    frame_to_show=frame_display_right,
                )

                # ROI / 调试输出（严格按 recognition_debug 开关）
                self._maybe_debug_outputs(
                    frame_id=frame_id,
                    left_image=left_image,
                    right_image=right_image,
                    debug_2d=debug_2d,
                    frame_display_left=frame_display_left,
                    frame_display_right=frame_display_right,
                    w_l=w_l,
                    h_l=h_l,
                    display_w_l=display_w_l,
                    display_h_l=display_h_l,
                    w_r=w_r,
                    h_r=h_r,
                    display_w_r=display_w_r,
                    display_h_r=display_h_r,
                    recognition_debug=recognition_debug,
                )

                if not success:
                    raise RecognitionWarning(
                        "双目识别失败",
                        "FACE_DETECTION_FAILED",
                        sleep_duration=self.interval,
                    )
                logger.debug("双目识别成功")
                # 坐标 debug：拟合点 3D (mm) 与可见性
                for eye in EYE_TYPE:
                    for fitting_type in FITTING_TYPE:
                        points = key_coordinates.get_points(eye, fitting_type)
                        for idx, pt in enumerate(points):
                            logger.debug(
                                "坐标 frame_id=%s eye=%s type=%s idx=%d x=%.3f y=%.3f z=%.3f v=%.3f",
                                frame_id, eye, fitting_type, idx, pt.x, pt.y, pt.z, pt.visibility,
                            )

                debug_2d = self.stereo_recognizer.get_last_debug_2d()
                # Left eye native from left image (debug_ll); right eye native from right image (debug_rr)
                native_geometry = {
                    "left": debug_2d.get("left", {}).get("debug", {}).get("native_geometry", {}),
                    "right": debug_2d.get("right", {}).get("debug_right_eye", {}).get("native_geometry", {}),
                }
                recg_fit_data_manager = RecgFitDataManager(
                    key_coordinates, debug_log=True, native_geometry=native_geometry
                )
                self.data_pipeline.store_recognition_data(frame_id, recg_fit_data_manager)
                self._cycle_update()
                
            except KeyboardInterrupt:
                raise KeyboardInterrupt
                
            except RecognitionWarning as w:
                # 处理其他警告（如帧读取失败等），但继续循环
                self.error_handler.handle_warning(w)
                continue
            except RecognitionError as e:
                logger.error(f"识别模块严重错误: {e}", exc_info=True)
                if not self.error_handler.handle_error(e):
                    self.running = False
                    break
            except Exception as e:
                logger.error(f"识别循环中发生未预期的异常: {e}", exc_info=True)
                # 继续循环，不终止
                continue
        # 识别线程退出时清理资源（在识别线程中执行，避免阻塞事件处理器）
        logger.info("识别线程主循环停止")
        # 关闭资源
        self.camera_manager.release_camera()
        cv2.destroyAllWindows()
        # 关闭 jsonl 句柄
        for f in self._roi_debug_files.values():
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        logger.info("OpenCV窗口已关闭")
            
    def _on_system_stop(self):
        """系统停止事件处理 - 只设置标志，资源清理在识别线程退出时执行"""
        self.running = False
        # 只注销事件处理器，资源清理在识别线程退出时执行（避免阻塞事件处理链）
        self._cleanup_event_handlers()
        logger.info("识别模块停止标志已设置（资源清理将在识别线程退出时执行）")
    
    def restart(self):
        """重启识别"""
        self.running = True
        self.run()
        
    def _cycle_update(self) -> None:
        """循环更新 - 发送事件而不是信号量"""
        self.error_handler.reset_warnings()
        frame_id = self.frame_id_manager.get_recognizing_frame_id()
        self.frame_id_manager.add_recognized_frame_id(frame_id)
        # 发送识别完成事件（非阻塞）
        CALLBACK_MANAGER.emit(RECOGNITION_COMPLETE, frame_id)
        # 使用可中断的sleep，检查running标志
        elapsed = 0
        while elapsed < self.interval and self.running:
            sleep(min(0.01, self.interval - elapsed))  # 每10ms检查一次
            elapsed += 0.01
        
    def _cycle_start(self) -> int:
        """循环开始"""
        # 每轮循环都生成新的 frame_id，避免失败路径复用同一 id
        self.frame_id_manager.generate_recognizing_frame_id()
        return self.frame_id_manager.get_recognizing_frame_id()

    def _maybe_debug_outputs(
        self,
        *,
        frame_id: int,
        left_image: BGRImage,
        right_image: BGRImage,
        debug_2d: Dict[str, Any],
        frame_display_left,
        frame_display_right,
        w_l: int,
        h_l: int,
        display_w_l: int,
        display_h_l: int,
        w_r: int,
        h_r: int,
        display_w_r: int,
        display_h_r: int,
        recognition_debug: bool,
    ) -> None:
        """
        将 run 里所有 debug 相关输出/落盘逻辑收口到一个入口。

        recognition_debug=False 时，函数直接返回，避免 jsonl/crop/overlay 落盘等行为。
        """
        if not recognition_debug:
            return

        # 兼容历史调试字段：保留原先 landmarks 的计算路径（当前代码没有后续使用）。
        left_points_dict = debug_2d.get("left", {}).get("points", {})
        landmarks_2d_left = []
        for eye_dict in left_points_dict.values():
            if not isinstance(eye_dict, dict):
                continue
            for pts in eye_dict.values():
                for p in pts:
                    landmarks_2d_left.append(
                        Landmark(x=p.x, y=p.y, z=0.0, visibility=1.0)
                    )
        if landmarks_2d_left:
            _ = scale_landmarks_for_display(
                landmarks_2d_left, w_l, h_l, display_w_l, display_h_l
            )

        right_points_dict = debug_2d.get("right", {}).get("points", {})
        landmarks_2d_right = []
        for eye_dict in right_points_dict.values():
            if not isinstance(eye_dict, dict):
                continue
            for pts in eye_dict.values():
                for p in pts:
                    landmarks_2d_right.append(
                        Landmark(x=p.x, y=p.y, z=0.0, visibility=1.0)
                    )
        if landmarks_2d_right:
            _ = scale_landmarks_for_display(
                landmarks_2d_right, w_r, h_r, display_w_r, display_h_r
            )

        # ROI 结构化 debug：jsonl + 轻量日志 + 额外 crop/overlay 图
        self._handle_roi_debug(
            frame_id=frame_id,
            frame_left_bgr=left_image.data,
            frame_right_bgr=right_image.data,
            debug_2d=debug_2d,
            frame_display_left=frame_display_left,
            frame_display_right=frame_display_right,
            recognition_debug=True,
        )

    # ------------------------------------------------------------------
    # ROI debug helpers
    # ------------------------------------------------------------------

    def _ensure_roi_debug_files(self) -> None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "debug",
            "roi",
        )
        os.makedirs(base_dir, exist_ok=True)
        for eye in ("left", "right"):
            if self._roi_debug_files[eye] is None:
                path = os.path.join(base_dir, f"roi_debug_{eye}.jsonl")
                # 使用 utf-8 追加模式
                self._roi_debug_files[eye] = open(path, "a", encoding="utf-8")

    def _handle_roi_debug(
        self,
        frame_id: int,
        frame_left_bgr,
        frame_right_bgr,
        debug_2d: Dict[str, Any],
        frame_display_left,
        frame_display_right,
        recognition_debug: bool,
    ) -> None:
        """统一处理每帧 ROI 的日志/jsonl/图片输出。"""
        try:
            self._ensure_roi_debug_files()
        except Exception as e:
            logger.error("初始化 ROI debug 文件失败: %s", e, exc_info=True)
            return

        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "debug",
            "roi",
        )

        for eye, disp_frame, side_key in (
            ("left", frame_display_left, "left"),
            ("right", frame_display_right, "right"),
        ):
            side_dbg = debug_2d.get(side_key, {})
            # 左眼使用 primary debug；右眼使用 debug_right_eye 分支
            if eye == "left":
                eye_dbg = side_dbg.get("debug", {})
            else:
                eye_dbg = side_dbg.get("debug_right_eye", {}) or side_dbg.get("debug", {})
            roi_debug_dict = eye_dbg.get("roi_debug") or {}

            # minimal ROI debug summary (aligned with roi_debug_analyzer columns)
            mode = str(roi_debug_dict["current_mode"])
            cg = float(roi_debug_dict["geometry_confidence"])
            face_stable = bool(roi_debug_dict["face_bbox_is_stable"])
            shift = float(roi_debug_dict["face_bbox_corner_max_shift"])
            track_from_previous_bbox_applied = bool(roi_debug_dict["track_from_previous_bbox_applied"])
            face_bbox = roi_debug_dict["face_bbox"]
            roi = roi_debug_dict["final_roi"]
            union_bbox = roi_debug_dict["union_mask_bbox"]
            logger.info(
                "ROI_DEBUG mode=%s cg=%.3f face_stable=%s shift=%.4f track_from_previous_bbox_applied=%s face_bbox=%s roi=%s union_bbox=%s",
                mode or "?",
                cg,
                face_stable,
                shift,
                track_from_previous_bbox_applied,
                face_bbox,
                roi,
                union_bbox,
            )

            # 结构化 jsonl
            try:
                f = self._roi_debug_files[eye]
                if f is not None and roi_debug_dict:
                    f.write(json.dumps(roi_debug_dict, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error("写入 ROI jsonl 失败 eye=%s: %s", eye, e, exc_info=True)

            # 额外保存 crop 与 crop_overlay
            if not recognition_debug or not roi_debug_dict:
                continue

            try:
                final_roi = roi_debug_dict["final_roi"]
                if final_roi is None:
                    continue
                x, y, w, h = final_roi
                source_frame = frame_left_bgr if eye == "left" else frame_right_bgr
                h_img, w_img = source_frame.shape[:2]
                x = max(0, min(int(x), w_img - 1))
                y = max(0, min(int(y), h_img - 1))
                w = max(1, min(int(w), w_img - x))
                h = max(1, min(int(h), h_img - y))
                crop = source_frame[y : y + h, x : x + w].copy()

                # base per-eye dirs
                eye_dir = os.path.join(base_dir, eye)
                crop_dir = os.path.join(eye_dir, "crop")
                overlay_dir = os.path.join(eye_dir, "crop_overlay")
                mask_color_dir = os.path.join(eye_dir, "mask_color")
                os.makedirs(crop_dir, exist_ok=True)
                os.makedirs(overlay_dir, exist_ok=True)
                os.makedirs(mask_color_dir, exist_ok=True)

                fname_base = (
                    f"frame_{frame_id}_{eye}_mode-{mode}_roi-{x}_{y}_{w}_{h}_cg-{cg:.2f}"
                )

                # 保存 crop 原图
                cv2.imwrite(os.path.join(crop_dir, fname_base + ".jpg"), crop)

                # EllSeg 分割 mask 彩图（与 crop 同尺寸对齐）
                native = eye_dbg.get("native_geometry", {})
                try:
                    mask_bgr = ellseg_segmentation_mask_to_bgr(
                        native.get("segmentation_mask"),
                        h,
                        w,
                    )
                    if mask_bgr is not None:
                        cv2.imwrite(
                            os.path.join(mask_color_dir, fname_base + ".jpg"),
                            mask_bgr,
                        )
                except Exception as mask_e:
                    logger.debug("保存 mask 彩图失败 eye=%s: %s", eye, mask_e)

                # crop 上仅叠加 EllSeg 几何（两椭圆 + 瞳孔中心），不写 debug 文字（与窗口上的 mode/cg 等文案区分）
                crop_overlay = crop.copy()
                native_in_crop = native_geometry_for_roi_crop(native, x, y)
                crop_overlay = draw_ellseg_native_geometry(
                    crop_overlay,
                    debug_primary={"native_geometry": native_in_crop},
                    debug_secondary=None,
                    src_w=w,
                    src_h=h,
                    dst_w=w,
                    dst_h=h,
                )
                cv2.imwrite(os.path.join(overlay_dir, fname_base + ".jpg"), crop_overlay)
            except Exception as e:
                logger.error("保存 ROI crop/overlay 失败 eye=%s: %s", eye, e, exc_info=True)
        
if __name__ == "__main__":
    try:
        recognition_manager = RecognitionManager()
        recognition_manager.run(warmup_frames=20, max_frames=50, recognition_debug=True)
    except Exception as e:
        logger.error("RecognitionManager 运行失败: %s", e, exc_info=True)