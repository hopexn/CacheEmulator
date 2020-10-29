import os

# 项目根目录
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


# 转化成绝对路径
def abs_path(*path):
    global _project_root
    return os.path.join(_project_root, *path)
