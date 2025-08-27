#!/usr/bin/env python3
"""
RANSAC结果分析脚本
分析多次RANSAC测试中各点的最终统计信息
"""

import re
import numpy as np
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import logging
import os
from datetime import datetime

def setup_logging(output_file: str = None):
    """
    设置日志输出
    
    Args:
        output_file: 输出文件路径
    """
    # 创建static/data目录
    os.makedirs('static/data', exist_ok=True)
    
    # 如果没有指定输出文件，使用默认文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'static/data/ransac_analysis_{timestamp}.log'
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return "static/data/" + output_file

def parse_log_file(log_file_path: str) -> Dict[str, List[float]]:
    """
    解析日志文件，提取各点的距离信息
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        Dict[str, List[float]]: 各点类型的距离列表
    """
    point_distances = defaultdict(lambda: {'center': [], 'surface': []})
    
    # 使用更直接的方法解析
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 找到所有RANSAC统计行
        ransac_lines = []
        for line in lines:
            if '平均到中心距离=' in line and '平均到表面距离=' in line and '(出现' in line:
                ransac_lines.append(line.strip())
        
        if ransac_lines:
            logging.info(f"找到 {len(ransac_lines)} 个RANSAC统计记录")
            
            # 按点类型分组，收集所有迭代的统计结果
            all_stats = defaultdict(list)
            
            for line in ransac_lines:
                try:
                    # 使用正则表达式提取信息
                    import re
                    
                    # 匹配点类型
                    point_match = re.search(r'(\w+):\s+平均到中心距离=', line)
                    if not point_match:
                        continue
                    point_type = point_match.group(1)
                    
                    # 匹配平均到中心距离
                    center_match = re.search(r'平均到中心距离=([\d.-]+)', line)
                    if not center_match:
                        continue
                    center_mean = float(center_match.group(1))
                    
                    # 匹配平均到表面距离
                    surface_match = re.search(r'平均到表面距离=([\d.-]+)', line)
                    if not surface_match:
                        continue
                    surface_mean = float(surface_match.group(1))
                    
                    # 匹配出现次数
                    count_match = re.search(r'出现(\d+)次', line)
                    if not count_match:
                        continue
                    count = int(count_match.group(1))
                    
                    # 收集所有迭代的统计结果
                    all_stats[point_type].append({
                        'center_mean': center_mean,
                        'surface_mean': surface_mean,
                        'count': count
                    })
                    
                except (ValueError, IndexError) as e:
                    logging.warning(f"解析错误: {line[:50]}..., 错误: {e}")
                    continue
            
            # 将所有迭代的统计结果添加到数据中
            for point_type, stats_list in all_stats.items():
                for stats in stats_list:
                    # 根据出现次数生成数据点
                    for _ in range(stats['count']):
                        point_distances[point_type]['center'].append(stats['center_mean'])
                        point_distances[point_type]['surface'].append(stats['surface_mean'])
    
    return point_distances

def calculate_statistics(point_distances: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    计算各点类型的统计信息
    
    Args:
        point_distances: 各点类型的距离数据
        
    Returns:
        Dict: 统计结果
    """
    stats = {}
    
    for point_type, distances in point_distances.items():
        center_dists = distances['center']
        surface_dists = distances['surface']
        
        if center_dists:  # 确保有数据
            stats[point_type] = {
                'center': {
                    'mean': np.mean(center_dists),
                    'min': np.min(center_dists),
                    'max': np.max(center_dists),
                    'std': np.std(center_dists),
                    'count': len(center_dists)
                },
                'surface': {
                    'mean': np.mean(surface_dists),
                    'min': np.min(surface_dists),
                    'max': np.max(surface_dists),
                    'std': np.std(surface_dists),
                    'count': len(surface_dists)
                }
            }
    
    return stats

def calculate_confidence_interval(data=None, mean=None, std=None, n=None, confidence_level=0.90):
    """
    计算置信区间（使用T分布）
    
    参数:
    data: 原始数据数组（可选）
    mean: 样本均值（如果提供数据，则忽略此参数）
    std: 样本标准差（如果提供数据，则忽略此参数）
    n: 样本大小（如果提供数据，则忽略此参数）
    confidence_level: 置信水平，默认为0.90（90%）
    
    返回:
    包含置信区间下限、上限和误差范围的元组
    """
    # 如果有原始数据，优先使用原始数据计算统计量
    if data is not None:
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # 使用无偏估计 (n-1)
    
    # 检查必要的参数是否提供
    if mean is None or std is None or n is None:
        raise ValueError("必须提供数据或均值、标准差和样本大小")
    
    # 计算自由度
    df = n - 1
    
    # 计算alpha值
    alpha = 1 - confidence_level
    
    # 计算标准误差
    se = std / np.sqrt(n)
    
    # 计算T分布的临界值（双尾）
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # 计算置信区间
    margin_of_error = t_critical * se
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return ci_lower, ci_upper, margin_of_error, t_critical, se

def log_statistics(stats: Dict[str, Dict[str, Dict[str, float]]], title: str = "RANSAC测试结果统计"):
    """
    将统计结果输出到日志
    
    Args:
        stats: 统计结果
        title: 标题
    """
    logging.info("=" * 80)
    logging.info(f"{title}")
    logging.info("=" * 80)
    
    if not stats:
        logging.warning("未找到有效的统计数据")
        return
    
    # 按点类型排序
    point_types = sorted(stats.keys())
    
    for point_type in point_types:
        point_stats = stats[point_type]
        center_stats = point_stats['center']
        surface_stats = point_stats['surface']
        
        logging.info(f"\n【{point_type.upper()}】点统计 (出现{center_stats['count']}次):")
        logging.info("-" * 60)
        
        # 计算到中心距离的置信区间
        ci_lower, ci_upper, moe, t_critical, se = calculate_confidence_interval(
            mean=center_stats['mean'], std=center_stats['std'], n=center_stats['count'], confidence_level=0.90
        )
        logging.info(f"  到中心距离:")
        logging.info(f"    平均值: {center_stats['mean']:.4f} mm")
        logging.info(f"    最小值: {center_stats['min']:.4f} mm")
        logging.info(f"    最大值: {center_stats['max']:.4f} mm")
        logging.info(f"    标准差: {center_stats['std']:.4f} mm")
        logging.info(f"    范围:   {center_stats['max'] - center_stats['min']:.4f} mm")
        logging.info(f"    标准误差: {se:.6f} mm")
        logging.info(f"    90%置信区间: ({ci_lower:.6f}, {ci_upper:.6f}) mm")
        logging.info(f"    误差范围: ±{moe:.6f} mm")
        
        # 计算到表面距离的置信区间
        ci_lower, ci_upper, moe, t_critical, se = calculate_confidence_interval(
            mean=surface_stats['mean'], std=surface_stats['std'], n=surface_stats['count'], confidence_level=0.90
        )
        logging.info(f"  到表面距离:")
        logging.info(f"    平均值: {surface_stats['mean']:.4f} mm")
        logging.info(f"    最小值: {surface_stats['min']:.4f} mm")
        logging.info(f"    最大值: {surface_stats['max']:.4f} mm")
        logging.info(f"    标准差: {surface_stats['std']:.4f} mm")
        logging.info(f"    范围:   {surface_stats['max'] - surface_stats['min']:.4f} mm")
        logging.info(f"    标准误差: {se:.6f} mm")
        logging.info(f"    90%置信区间: ({ci_lower:.6f}, {ci_upper:.6f}) mm")
        logging.info(f"    误差范围: ±{moe:.6f} mm")
    
    # 计算整体统计
    logging.info(f"\n【整体统计】:")
    logging.info("-" * 60)
    
    all_center_dists = []
    all_surface_dists = []
    
    for point_type in point_types:
        all_center_dists.extend([stats[point_type]['center']['mean']] * stats[point_type]['center']['count'])
        all_surface_dists.extend([stats[point_type]['surface']['mean']] * stats[point_type]['surface']['count'])
    
    if all_center_dists:
        overall_avg_center = np.mean(all_center_dists)
        overall_avg_surface = np.mean(all_surface_dists)
        overall_min_center = np.min(all_center_dists)
        overall_max_center = np.max(all_center_dists)
        overall_min_surface = np.min(all_surface_dists)
        overall_max_surface = np.max(all_surface_dists)
        overall_std_center = np.std(all_center_dists)
        overall_std_surface = np.std(all_surface_dists)
        
        # 计算整体到中心距离的置信区间
        ci_lower, ci_upper, moe, t_critical, se = calculate_confidence_interval(
            mean=overall_avg_center, std=overall_std_center, n=len(all_center_dists), confidence_level=0.90
        )
        logging.info(f"  到中心距离:")
        logging.info(f"    平均值: {overall_avg_center:.4f} mm")
        logging.info(f"    最小值: {overall_min_center:.4f} mm")
        logging.info(f"    最大值: {overall_max_center:.4f} mm")
        logging.info(f"    标准差: {overall_std_center:.4f} mm")
        logging.info(f"    范围:   {overall_max_center - overall_min_center:.4f} mm")
        logging.info(f"    标准误差: {se:.6f} mm")
        logging.info(f"    90%置信区间: ({ci_lower:.6f}, {ci_upper:.6f}) mm")
        logging.info(f"    误差范围: ±{moe:.6f} mm")
        
        # 计算整体到表面距离的置信区间
        ci_lower, ci_upper, moe, t_critical, se = calculate_confidence_interval(
            mean=overall_avg_surface, std=overall_std_surface, n=len(all_surface_dists), confidence_level=0.90
        )
        logging.info(f"  到表面距离:")
        logging.info(f"    平均值: {overall_avg_surface:.4f} mm")
        logging.info(f"    最小值: {overall_min_surface:.4f} mm")
        logging.info(f"    最大值: {overall_max_surface:.4f} mm")
        logging.info(f"    标准差: {overall_std_surface:.4f} mm")
        logging.info(f"    范围:   {overall_max_surface - overall_min_surface:.4f} mm")
        logging.info(f"    标准误差: {se:.6f} mm")
        logging.info(f"    90%置信区间: ({ci_lower:.6f}, {ci_upper:.6f}) mm")
        logging.info(f"    误差范围: ±{moe:.6f} mm")

def analyze_multiple_logs(log_files: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    分析多个日志文件
    
    Args:
        log_files: 日志文件路径列表
        
    Returns:
        Dict: 合并后的统计结果
    """
    all_point_distances = defaultdict(lambda: {'center': [], 'surface': []})
    
    for log_file in log_files:
        logging.info(f"正在分析日志文件: {log_file}")
        point_distances = parse_log_file(log_file)
        
        # 合并数据
        for point_type, distances in point_distances.items():
            all_point_distances[point_type]['center'].extend(distances['center'])
            all_point_distances[point_type]['surface'].extend(distances['surface'])
    
    return calculate_statistics(all_point_distances)

def main():
    parser = argparse.ArgumentParser(description='分析RANSAC测试结果')
    parser.add_argument('log_files', nargs='+', help='日志文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 先设置日志输出
    output_file = setup_logging(args.output)
    logging.info(f"分析结果将保存到: {output_file}")
    
    # 处理通配符
    import glob
    expanded_files = []
    for pattern in args.log_files:
        if '*' in pattern or '?' in pattern:
            # 处理通配符
            matched_files = glob.glob(pattern)
            if matched_files:
                expanded_files.extend(matched_files)
            else:
                logging.warning(f"未找到匹配的文件: {pattern}")
        else:
            # 直接添加文件路径
            expanded_files.append(pattern)
    
    if not expanded_files:
        logging.error("没有找到有效的日志文件")
        return
    
    # 去重并排序
    expanded_files = sorted(list(set(expanded_files)))
    logging.info(f"找到 {len(expanded_files)} 个日志文件: {expanded_files}")
    
    # 分析日志文件
    if len(expanded_files) == 1:
        logging.info(f"分析单个日志文件: {expanded_files[0]}")
        point_distances = parse_log_file(expanded_files[0])
        stats = calculate_statistics(point_distances)
        log_statistics(stats, f"RANSAC测试结果统计 - {expanded_files[0]}")
    else:
        logging.info(f"分析多个日志文件: {len(expanded_files)} 个文件")
        stats = analyze_multiple_logs(expanded_files)
        log_statistics(stats, f"RANSAC测试结果统计 - 合并{len(expanded_files)}个文件")
    
    logging.info(f"分析完成，结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
    
'''
# 分析单个日志文件
python analyze_ransac_results.py logs/recognition_20250827_215034.log

# 分析多个日志文件
python analyze_ransac_results.py logs/recognition_20250827_215034.log logs/recognition_20250827_225404.log

# 使用通配符分析所有日志文件
python analyze_ransac_results.py logs/*.log

# 指定输出文件
python analyze_ransac_results.py logs/recognition_20250827_215034.log -o my_analysis.log


'''