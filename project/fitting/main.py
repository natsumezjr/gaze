import numpy as np
from typing import Dict, List, Union

def main(key_coordinates: dict):
    """
    眼动追踪数据初步测评
    """
    print("眼动数据测评:")
    
    # 瞳孔中心
    if 'pupil_centers' in key_coordinates:
        pupil = key_coordinates['pupil_centers']
        if pupil['left'] is not None and pupil['right'] is not None:
            left = pupil['left']
            right = pupil['right']
            print(f"瞳孔: 左({left[0]}, {left[1]}, {left[2]})\n右({right[0]}, {right[1]}, {right[2]})")
        else:
            print("瞳孔: 无效")
    


if __name__ == "__main__":
    main({})