# 人脸关键点提取器使用示例

## 函数列表

### 1. extract_landmarks(rgb_image)
提取468个人脸关键点

### 2. validate_landmarks(landmarks)
验证关键点质量

### 3. get_eye_landmarks(landmarks, eye_type)
提取指定眼睛的关键点

### 4. get_pupil_landmarks(landmarks)
提取左右瞳孔中心关键点

### 5. get_iris_landmarks(landmarks)
提取左右虹膜边界关键点

### 6. get_landmark_indices()
获取关键点索引映射

## 使用示例

### 基础使用流程

```python
import cv2
import numpy as np
from core.landmark_extractor import (
    extract_landmarks, 
    validate_landmarks, 
    get_eye_landmarks, 
    get_pupil_landmarks, 
    get_iris_landmarks
)

# 1. 读取图像
image = cv2.imread("face_image.jpg")

# 2. 提取关键点
landmarks = extract_landmarks(image)

if landmarks:
    # 3. 验证关键点质量
    if validate_landmarks(landmarks):
        print("关键点质量良好")
        
        # 4. 提取眼部关键点
        left_eye = get_eye_landmarks(landmarks, 'left')
        right_eye = get_eye_landmarks(landmarks, 'right')
        
        # 5. 提取瞳孔中心
        pupil_center = get_pupil_landmarks(landmarks)
        
        # 6. 提取虹膜边界
        iris_boundaries = get_iris_landmarks(landmarks)
        
        print(f"左眼关键点数量: {len(left_eye)}")
        print(f"右眼关键点数量: {len(right_eye)}")
        print(f"左瞳孔中心: {pupil_center['left']}")
        print(f"右瞳孔中心: {pupil_center['right']}")
    else:
        print("关键点质量较差")
else:
    print("未检测到人脸")
```

### 摄像头实时处理

```python
import cv2
from core.landmark_extractor import extract_landmarks, validate_landmarks

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 提取关键点
    landmarks = extract_landmarks(frame)
    
    if landmarks and validate_landmarks(landmarks):
        # 在图像上绘制关键点
        for x, y, z in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
        
        cv2.putText(frame, f"Landmarks: {len(landmarks)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Face Landmarks', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 关键点可视化

```python
import cv2
import numpy as np
from core.landmark_extractor import (
    extract_landmarks, 
    get_eye_landmarks, 
    get_pupil_landmarks, 
    get_iris_landmarks
)

def draw_landmarks(image, landmarks):
    """在图像上绘制关键点"""
    result = image.copy()
    
    # 绘制所有关键点（绿色小点）
    for x, y, z in landmarks:
        cv2.circle(result, (int(x), int(y)), 1, (0, 255, 0), -1)
    
    # 绘制左眼关键点（蓝色）
    left_eye = get_eye_landmarks(landmarks, 'left')
    for x, y, z in left_eye:
        cv2.circle(result, (int(x), int(y)), 3, (255, 0, 0), -1)
    
    # 绘制右眼关键点（红色）
    right_eye = get_eye_landmarks(landmarks, 'right')
    for x, y, z in right_eye:
        cv2.circle(result, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # 绘制瞳孔中心（黄色大点）
    pupil_center = get_pupil_landmarks(landmarks)
    if pupil_center['left']:
        x, y, z = pupil_center['left']
        cv2.circle(result, (int(x), int(y)), 5, (0, 255, 255), -1)
    if pupil_center['right']:
        x, y, z = pupil_center['right']
        cv2.circle(result, (int(x), int(y)), 5, (0, 255, 255), -1)
    
    return result

# 使用示例
image = cv2.imread("face_image.jpg")
landmarks = extract_landmarks(image)

if landmarks:
    visualized_image = draw_landmarks(image, landmarks)
    cv2.imshow('Face Landmarks', visualized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    cv2.imwrite('landmarks_result.jpg', visualized_image)
```

## 运行测试

```bash
# 运行完整测试
cd project/recognition
python test_all_functions.py

# 运行基础测试
python test_landmark_extractor.py
``` 