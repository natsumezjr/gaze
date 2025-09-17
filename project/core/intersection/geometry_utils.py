# 几何工具模块
from project.data.data_models import Vector3D
import numpy as np

def normalize_vector(p1, p2) -> Vector3D:
    # 自动处理两种类型
    if hasattr(p1, 'to_point3d'):
        p1 = p1.to_point3d()
    if hasattr(p2, 'to_point3d'):
        p2 = p2.to_point3d()
        
    if hasattr(p1, 'to_ndarray'):
        p1 = p1.to_ndarray()
    if hasattr(p2, 'to_ndarray'):
        p2 = p2.to_ndarray()
    
    return (p2 - p1) / np.linalg.norm(p2 - p1)

