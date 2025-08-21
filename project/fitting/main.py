import numpy as np
from typing import Dict, List, Union
from project.fitting.core.ransac_sphere_fitter import ransac_sphere_eyeball,ransac_sphere_eyeball_restart
from project.fitting.utils.geometry import center_fitter

def main(key_coordinates: dict):
    """
    眼动追踪数据初步测评
    """
    
    # 瞳孔中心
    if 'pupil_centers' in key_coordinates:
        pupil = key_coordinates['pupil_centers']
        if pupil['left'] is not None and pupil['right'] is not None:
            left = pupil['left']
            right = pupil['right']
            print(f"瞳孔: 左({left[0]}, {left[1]}, {left[2]})\n右({right[0]}, {right[1]}, {right[2]})")
        else:
            print("瞳孔: 无效")
    
    if 'iris_boundaries' in key_coordinates:
        iris = key_coordinates['iris_boundaries']
        if iris['left'] is not None and iris['right'] is not None:
            left = iris['left']
            right = iris['right']
            print(f"虹膜: 左({left[0]}, {left[1]}, {left[2]})\n右({right[0]}, {right[1]}, {right[2]})")
        else:
            print("虹膜: 无效")
            
    if 'eyes_contours' in key_coordinates:
        contour = key_coordinates['eyes_contours']
        if contour['left'] is not None and contour['right'] is not None:
            left = contour['left']
            right = contour['right']
            print(f"眼睛轮廓: 左({left[0]}, {left[1]}, {left[2]})\n右({right[0]}, {right[1]}, {right[2]})")
        else:
            print("眼睛轮廓: 无效")
            
    ransac_sphere_eyeball(key_coordinates,center_fitter)
    

if __name__ == "__main__":
    main({})