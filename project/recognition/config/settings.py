# 这个是全局的路径配置
DATA_PATH = "project/recognition/data"
CAMERA_PARAMS_PATH = "project/recognition/config/camera_params.json"


# 数据清理相关配置
MAX_AGE_SECONDS = 10  # 数据最大保留时间（秒）
CLEANUP_INTERVAL = 1   # 清理线程检查间隔（秒）

# 文件格式配置
DATA_FILE_FORMAT = "pkl"  # 数据文件格式：pkl 或 json
DATA_FILENAME = f"rgbd_input.{DATA_FILE_FORMAT}"  # 数据文件名

# 检测阈值
DETECTION_CONFIDENCE_THRESHOLD = 0.7
DEPTH_VALIDATION_THRESHOLD = 0.1
COORDINATE_QUALITY_THRESHOLD = 0.8
MAX_PROCESSING_TIME = 0.1  # 秒