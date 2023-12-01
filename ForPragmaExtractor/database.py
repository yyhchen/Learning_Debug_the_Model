import sys
sys.path.append("..")
import os
import json
import shutil
import ForPragmaExtractor.global_parameters as gp
import pickle

class Database:
    def __init__(self, path, json_path,  override=False):
        if override:
            inp = 'y'
            if inp == 'y':
                if os.path.isdir(path):
                    shutil.rmtree(path)     # 递归删除当前目录下的所有文件夹(除了自身)
                if os.path.isfile(json_path):
                    os.remove(json_path)    # 删除指定文件
                print("Removing", path)
            else:
                print("You chose no, continue as if not inserted override...")

        if not os.path.isdir(path):
            os.mkdir(path)
        self.db_path = path     # 存储项目文件路径
        if not os.path.isfile(json_path):   # 如果指定路径没有文件，就创一个
            with open(json_path, 'w') as f:
                dict1 = {"key": 1}
                print(dict1)
                json.dump(dict1, f, indent=4)     # ident是为了缩进; f是文件对象，存储序列化的json数据
        self.json_path = json_path  # 存储项目信息的json文件
        self.override = override


    def insert(self, pragmafor, project_name, id=0):
#def insert(self, pragmafor: gp.PragmaForTuple, project_name: str, id=0):
        '''
        insert方法用于将一个新的项目添加到数据库中。
        它接受三个参数：pragmafor（项目中的 pragma 信息），project_name（项目名称），id（项目ID，默认为0）。
        在添加项目时，会创建一个项目文件夹，并将 pragma 信息、代码片段（分为不使用 OpenMP 和使用 OpenMP 两种情况）
        以及项目的原始坐标等信息存储在其中。同时，将项目信息添加到 JSON 文件中。
        '''
        folder_path = os.path.join(self.db_path, project_name)
        file_data_pragma, file_data_for_loops = pragmafor.get_string_data()  # pragma 和 code snippet
        original_file = pragmafor.get_coord()   # 获取坐标
        if self.check_if_repo_exists(project_name):
            print("Folder:", folder_path, " exists")
            return

        # As a part of the database, we create a folder of the project
        os.mkdir(folder_path)

        # Now we copy the pragma.c and code.c that contains the relevant code segments.
        no_openmp = os.path.join(folder_path, gp.FULL_CODE_NAME + ".c")
        pickle_file = os.path.join(folder_path, gp.PICKLE_CODE_NAME + ".pkl")
        with_openmp = os.path.join(folder_path, gp.OPENMP_CODE_NAME + ".c")

        # CREATE AND WRITE THE FILES
        if file_data_for_loops != "":
            f = open(no_openmp, "w")
            f.writelines(file_data_for_loops)
            f.close()
        else:
            print("NO FOR LOOP DATA")
            input()
            no_openmp = ""
        if file_data_pragma != "":
            f = open(with_openmp, "w")
            f.writelines(file_data_pragma)
            f.close()
        else:
            with_openmp = ""
        pickle.dump(pragmafor, open(pickle_file, "wb"), protocol = 2)
        # Now we add project_name to the json (should be the username along with the project name)
        # And add to the that key, the path to pragma.c and code.c
        self._write_json(project_name, gp.FULL_CODE_NAME, no_openmp)
        self._write_json(project_name, gp.OPENMP_CODE_NAME, with_openmp)
        self._write_json(project_name, gp.PICKLE_CODE_NAME, pickle_file)
        self._write_json(project_name, "original", original_file)
        self._write_json(project_name, "id", id)

    def _write_json(self, proj_name, key, new_data):
        filename = self.json_path
        with open(filename, 'r+') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            if proj_name in file_data:
                file_data[proj_name][key] = new_data
            else:
                file_data[proj_name] = {}
                file_data[proj_name][key] = new_data
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent = 4)

    def _exists_in_json(self, key1, key2=""):
        filename = self.json_path
        with open(filename, 'r+') as file:
            # First we load existing data into a dict.
            try:
                file_data = json.load(file)
                if key1 in file_data:
                    if not key2:
                        return True
                    return key2 in file_data[key1]
                else:
                    return False
            except:
                return False


    '''
        用于检查给定的项目名称（project_name）是否存在于 JSON 文件中
    '''
    def check_if_repo_exists(self, project_name):
        return self._exists_in_json(project_name) and \
               self._exists_in_json(project_name, gp.FULL_CODE_NAME) and \
               self._exists_in_json(project_name, gp.OPENMP_CODE_NAME)

        # Join new_data with file_data inside emp_details
        # Sets file's current position at offset.
        # convert back to json.

