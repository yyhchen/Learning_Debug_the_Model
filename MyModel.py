import json
import pickle as pkl
from pycparser import c_generator
import ForPragmaExtractor.visitors as visitor
from Model import tokenizer

VAR_PREFIX = "var"
ARR_PREFIX = "arr"
FUNC_PREFIX = "func"
STRUCT_PREFIX = "struct"
generator = c_generator.CGenerator()
id_v = visitor.CounterIdVisitor()
replacer = visitor.ReplaceIdsVisitor(VAR_PREFIX, ARR_PREFIX, STRUCT_PREFIX, FUNC_PREFIX)


def db_read_string_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            return "".join(f.readlines())   # readlines()是读全部
    except:
        return ""


def normalize_code_as_ast(pickle_file):
    # print (pickle_file)
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
        # for_ast.show()
        # print(normalize_code_as_string.generator.visit(for_ast))
        # for_ast.show()
        # counts in an array the name and identifiers of the code
        id_v.reset()
        id_v.visit(for_ast)
        # Replace the names..
        replacer.reset(id_v.ids, id_v.array,id_v.struct, id_v.func)
        replacer.visit(for_ast)
        with open('temp.txt', 'w') as f:
            for_ast.show(buf=f)
        with open('temp.txt', 'r') as f:
            ast = f.readlines()

        ast_no_whitespaces = [a.strip() for a in ast] # kill all whitespaces
        # print(ast_no_whitespaces)
        # print(normalize_code_as_string.generator.visit(for_ast))

        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        ast_one_line = " " + " ".join(ast_no_whitespaces)
        return ast_one_line


# str1 = normalize_code_as_ast('DB_TEST/DB_TEST/database/PolyBench-ACC-master_.gitignore_2mm.c_5/code_pickle.pkl')
# print(str1)

def code_as_ast(pickle_file):
    # print (pickle_file)
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
        with open('temp.txt', 'w') as f:
            for_ast.show(buf=f)
        with open('temp.txt', 'r') as f:
            ast = f.readlines()

        ast_no_whitespaces = [a.strip() for a in ast] # kill all whitespaces and \n
        ast_one_line = " " + " ".join(ast_no_whitespaces)
        return ast_one_line


str2 = code_as_ast('DB_TEST/DB_TEST/database/PolyBench-ACC-master_.gitignore_2mm.c_5/code_pickle.pkl')
print(str2)
# print('str2len: ', len(str2))

data1 = {'text':[str2]}

text, _ = tokenizer.deepscc_tokenizer(data1['text'])
print(text)
# print('tokenlen: ', len(text.input_ids[0]), len(text.attention_mask[0]))


# ------------------------------------------------------------------------

data_set = {'text':[], 'label':[], 'ast':[]}
jsonpath = 'D:\CodeLibrary\PragFormer-main\DB_TEST\DB_TEST\database.json'
with open(jsonpath, 'r') as f:
    file_data = json.load(f)

    for i, key in enumerate(file_data):
        print(file_data[key]["code"])
        code = db_read_string_from_file(file_data[key]["code"])
        ast_str = code_as_ast(file_data[key]['code_pickle'])
        if file_data[key]['pragma']:
            pragma = db_read_string_from_file(file_data[key]['pragma'])
        else:
            pragma = 0

        data_set['text'].append(code)
        data_set['ast'].append(ast_str)
        data_set['label'].append(pragma)



