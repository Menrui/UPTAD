import sys
import os


# print(os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib")))
src_dir_path=os.path.dirname(__file__)
workspace_dir_path=os.path.dirname(src_dir_path)
lib_dir_path=os.path.join(workspace_dir_path, "lib")
sys.path.append(os.path.join(lib_dir_path, "pytorch-ssim"))
sys.path.append(os.path.join(lib_dir_path, "partialconv"))