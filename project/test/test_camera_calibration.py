import os
from camera_calibration import (
    load_camera_params,
    save_camera_params,
    validate_camera_params,
    get_intrinsic_matrix,
    get_camera_info
)

def test_camera_calibration():
    # 1. 创建一个测试参数字典
    test_params = {
        "fx": 1000.0,
        "fy": 1000.0,
        "cx": 640.0,
        "cy": 360.0
    }

    # 2. 定义测试文件路径
    test_file = "test_camera_params.json"

    # 3. 保存参数到文件
    save_camera_params(test_params, test_file)
    print("[✔] 参数保存成功")

    # 4. 从文件加载参数
    loaded_params = load_camera_params(test_file)
    print("[✔] 参数加载成功")

    # 5. 验证参数
    if validate_camera_params(loaded_params):
        print("[✔] 参数验证通过")
    else:
        print("[✘] 参数验证失败")
        return

    # 6. 获取内参矩阵
    K = get_intrinsic_matrix(loaded_params)
    print("[✔] 内参矩阵 K：")
    print(K)

    # 7. 获取信息字符串
    info = get_camera_info(loaded_params)
    print("[✔] 相机信息：")
    print(info)

    # 8. 清理测试文件
    os.remove(test_file)
    print("[✔] 测试文件已删除")

if __name__ == "__main__":
    test_camera_calibration()
