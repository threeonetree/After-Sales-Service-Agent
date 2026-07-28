'''
为整个工程提供统一的绝对路径
'''
import os

def get_project_root()->str:
    """
    获取项目根目录
    先获得文件绝对路径，再往上倒两层即可获得
    :return:
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_abs_path(relative_path: str)->str:
    """
    获取绝对路径
    :param relative_path: 相对路径
    把获得的根目录和传进来的相对路径组合得到其绝对路径
    :return:
    """
    return os.path.join(get_project_root(), relative_path)

if __name__ == '__main__':
    print(get_project_root())
    print(get_abs_path('data/data.csv'))