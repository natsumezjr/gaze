
import logging
from project.fitting.core.fitting_strategy import fit_all_eyes
from project.recg_fit_data.data_manager import RECG_FIT_DATA_MANAGER
def main():
    """
    眼动追踪数据初步测评
    """
    #key_coordinates_log(RECG_FIT_DATA_MANAGER)
    
    
    logging.info("-" * 50)
    logging.info("开始RANSAC球体拟合...")
    
    results = fit_all_eyes(params_only=False)

    data_manager = RECG_FIT_DATA_MANAGER
    data_manager._debug_log = True
    data_manager._log_debug()
    
    for key, result in results.items():
        logging.info(f"拟合结果: {key}：\n{result}")
        
    
        
        
    
    # 适当睡眠防止刷屏
    

if __name__ == "__main__":
    # 创建空的RecgFitDataManager实例用于测试
    main()