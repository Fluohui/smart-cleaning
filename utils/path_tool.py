"""
为整个工程提供统一的绝对路径
"""

import os

def get_project_root() -> str:
    """
    获取工程所在的根目录
    :return: 字符串根目录
    """
    # 当前文件所在的目录
    current_file = os.path.realpath(__file__)
    # 当前文件所在的目录的父目录
    current_dir = os.path.dirname(current_file)
    # 工程所在的根目录
    project_root = os.path.dirname(current_dir)

    return project_root

def get_abs_path(relative_path: str) -> str:
    """
    传递相对路径，得到绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    # 获取工程根目录
    project_root = get_project_root()
    # 获取绝对路径
    abs_path = os.path.join(project_root, relative_path)

    return abs_path