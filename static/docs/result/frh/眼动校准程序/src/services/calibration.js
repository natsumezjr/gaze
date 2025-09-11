/**
 * 校准反馈接口
 * 用于处理校准结果和kappa角补偿
 */

/**
 * 校准结果接口
 * @typedef {Object} CalibrationResult
 * @property {string} status - 校准状态 ('success' | 'error')
 * @property {string} model_id - 校准模型ID
 * @property {number} accuracy - 校准精度（百分比）
 * @property {Object} kappa_params - kappa角参数
 * @property {Object} kappa_params.left_eye - 左眼kappa角参数
 * @property {Array<number>} kappa_params.left_eye.axis - 旋转轴 [x, y, z]
 * @property {number} kappa_params.left_eye.angle - 旋转角度（弧度）
 * @property {Object} kappa_params.right_eye - 右眼kappa角参数
 * @property {Array<number>} kappa_params.right_eye.axis - 旋转轴 [x, y, z]
 * @property {number} kappa_params.right_eye.angle - 旋转角度（弧度）
 * @property {number} samples_processed - 处理的样本数量
 */

// 存储当前的校准模型
let currentCalibrationModel = null;

/**
 * 应用kappa角补偿
 * @param {Object} gazePoint - 原始视线点 {x, y}
 * @param {Object} kappaParams - kappa角参数
 * @returns {Object} 补偿后的视线点 {x, y}
 */
export function applyKappaCompensation(gazePoint, kappaParams) {
  // 预留接口，由用户自行实现kappa角补偿算法
  console.log('应用kappa角补偿', gazePoint, kappaParams);
  return { ...gazePoint };
}

/**
 * 计算校准精度
 * @param {Array} calibrationData - 校准数据
 * @returns {number} 校准精度（百分比）
 */
export function calculateCalibrationAccuracy(calibrationData) {
  // 预留接口，由用户自行实现校准精度计算算法
  console.log('计算校准精度', calibrationData);
  return 95.0;
}

/**
 * 保存校准模型
 * @param {Object} calibrationResult - 校准结果
 * @returns {Promise<boolean>} 是否保存成功
 */
export async function saveCalibrationModel(calibrationResult) {
  try {
    // 保存校准模型到全局变量
    currentCalibrationModel = calibrationResult;
    
    // 如果有localStorage，也可以保存到本地存储
    if (window.localStorage) {
      localStorage.setItem('calibrationModel', JSON.stringify(calibrationResult));
    }
    
    console.log('校准模型已保存:', calibrationResult);
    return true;
  } catch (error) {
    console.error('保存校准模型失败:', error);
    return false;
  }
}

/**
 * 加载校准模型
 * @param {string} modelId - 校准模型ID
 * @returns {Promise<Object>} 校准模型
 */
export async function loadCalibrationModel(modelId) {
  try {
    // 如果有指定模型ID，尝试从服务器加载
    if (modelId) {
      // 这里应该实现从服务器加载模型的逻辑
      console.log('尝试从服务器加载模型:', modelId);
      // 暂时返回当前模型
      return currentCalibrationModel;
    }
    
    // 如果没有指定模型ID，尝试从本地存储加载
    if (window.localStorage) {
      const savedModel = localStorage.getItem('calibrationModel');
      if (savedModel) {
        return JSON.parse(savedModel);
      }
    }
    
    // 如果本地存储也没有，返回当前模型
    return currentCalibrationModel;
  } catch (error) {
    console.error('加载校准模型失败:', error);
    return null;
  }
}

/**
 * 应用校准模型进行视线修正
 * @param {Object} gazePoint - 原始视线点 {x, y}
 * @param {Object} calibrationModel - 校准模型
 * @returns {Object} 修正后的视线点 {x, y}
 */
export function applyCalibrationModel(gazePoint, calibrationModel) {
  if (!calibrationModel || !gazePoint) {
    return { ...gazePoint };
  }
  
  try {
    // 提取kappa角参数
    const kappaParams = calibrationModel.kappa_params;
    if (!kappaParams) {
      return { ...gazePoint };
    }
    
    // 应用kappa角补偿
    return applyKappaCompensation(gazePoint, kappaParams);
  } catch (error) {
    console.error('应用校准模型失败:', error);
    return { ...gazePoint };
  }
}

/**
 * 获取当前校准模型
 * @returns {Object} 当前校准模型
 */
export function getCurrentCalibrationModel() {
  return currentCalibrationModel;
}

/**
 * 应用当前校准模型进行视线修正
 * @param {Object} gazePoint - 原始视线点 {x, y}
 * @returns {Object} 修正后的视线点 {x, y}
 */
export function applyCurrentCalibrationModel(gazePoint) {
  return applyCalibrationModel(gazePoint, currentCalibrationModel);
}