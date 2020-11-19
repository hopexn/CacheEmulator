import os

# 项目根目录
_package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


# 转化成绝对路径
def relative_package_path(*path):
    global _package_root
    return os.path.join(_package_root, *path)
