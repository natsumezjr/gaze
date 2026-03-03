# 立体/深度相关常量（不在 camera_params.json 中的配置）
# image_size 等标定相关尺寸以 camera_params.json 为准，此处仅放运行时用到的常数。

# 视为“双目并排”的整帧分辨率列表 (width, height)，用于判断是否走立体深度流程
STEREO_FRAME_SIZES = [
    (3840, 1080),
    (2560, 1080),
]

# 无双目标定时占位深度（米），用于占位深度图
PLACEHOLDER_DEPTH_METERS = 0.4

# 无双目标定时默认深度缩放（camera_unit -> 米），与 detector 约定一致
DEFAULT_DEPTH_SCALE = 0.001

# StereoSGBM 立体匹配参数（numDisparities 须为 16 的整数倍）
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 16 * 16  # 256
SGBM_BLOCK_SIZE = 5
SGBM_P1 = 8 * 3 * 5 ** 2
SGBM_P2 = 32 * 3 * 5 ** 2
SGBM_DISP12_MAX_DIFF = 1
SGBM_UNIQUENESS_RATIO = 10
SGBM_SPECKLE_WINDOW_SIZE = 100
SGBM_SPECKLE_RANGE = 32
SGBM_PREFILTER_CAP = 63
