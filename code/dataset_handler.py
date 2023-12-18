import copy
import random
import string
import subprocess
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import ast
from lib2to3 import refactor
import re
import ast
import builtins
import os
import re



class CodeCleaner(ast.NodeTransformer):
    def __init__(self):
        self.var_counter = 1
        self.func_counter = 1
        self.name_mapping = {}
        self.func_name_mapping = {}
        self.imported_modules = set()
        self.builtin_names = dir(builtins)
        self.function_defs = [] # stores individual functions

    def visit_Import(self, node):
        for alias in node.names:
            self.imported_modules.add(alias.name)
        return node

    def visit_ImportFrom(self, node):
        self.imported_modules.add(node.module)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store)) and node.id not in self.builtin_names:
            if node.id not in self.name_mapping and node.id not in self.imported_modules:
                self.name_mapping[node.id] = f'var{self.var_counter}'
                self.var_counter += 1
            new_name = self.name_mapping.get(node.id, node.id)
        else:
            new_name = node.id
        return ast.copy_location(ast.Name(id=new_name, ctx=node.ctx), node)

    def visit_FunctionDef(self, node):
        original_name = node.name

        original_node = copy.deepcopy(node)
        original_node.name = original_name

        # Rename the function for further AST processing and mapping
        if node.name not in self.builtin_names:
            node.name = f'func{self.func_counter}'
            self.func_counter += 1
            self.name_mapping[original_name] = node.name
            self.func_name_mapping[original_name] = node.name

        self.function_defs.append(node)

        self.generic_visit(node)
        return node


    def visit_ClassDef(self, node):
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self.visit_FunctionDef(item)

        return None


def convert_python2_to_python3(code):
    tool = refactor.RefactoringTool(refactor.get_fixers_from_package('lib2to3.fixes'))
    tree = tool.refactor_string(code, '<string>')
    return str(tree)

def extract_functions(cleaner):
    function_sources = []
    for func in cleaner.function_defs:
        source = ast.unparse(func)
        function_sources.append(source)
    return function_sources


def clean_code(code):
    comments, code = extract_and_remove_comments(code)
    comments = None
    if not comments and not code:
        return None, None, {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        try:
            code = convert_python2_to_python3(code)
            tree = ast.parse(code)
        except Exception as e:
            print(f"Error converting code: {e}")
            return None, None, {}
    cleaner = CodeCleaner()
    tree = cleaner.visit(tree)

    try:
        cleaned_code = ast.unparse(tree)  # Return the source as a string
    except SyntaxError as e:
        print(f"SyntaxError after conversion: {e}")
        return None, None, {}
    
    function_sources = extract_functions(cleaner)
    if not function_sources:
        function_sources = [cleaned_code]
    return cleaned_code, comments, cleaner.name_mapping, function_sources

def extract_and_remove_comments(code):
    comments = re.findall(r'(?m)^\s*#.*$', code)

    def replace_with_whitespace(match):
        return ' ' * len(match.group(0))

    code_without_comments = re.sub(r'(?m)^\s*#.*$', replace_with_whitespace, code)
    return ' '.join(comments), code_without_comments

# TODO: we could also remove the imports if we want in this function
def rename_functions(function_list):
    renamed_functions = []

    for func in function_list:
        # Find the function name using a regex
        match = re.search(r'\bdef\s+(\w+)\s*\(', func)
        if match:
            original_name = match.group(1)
            # Replace the original function name with "func1", "func2", etc.
            renamed_func = re.sub(r'\bdef\s+' + original_name + r'\b', f'def func1', func, 1)
            renamed_functions.append((renamed_func.strip(), original_name))

    return renamed_functions

def process_dataset_item(code):
    code_content = code['content']
    comments, code_without_comments = extract_and_remove_comments(code_content)
    if not comments and not code_without_comments:
        return None

    try:
        original_tree = ast.parse(code_without_comments)
    except SyntaxError:
        # print('Syntax error in original code')
        return None

    cleaned_code, comments, name_mappings, function_sources = clean_code(code_without_comments)
    if function_sources is None:
        return []
    
    function_set = rename_functions(function_sources)


    # TODO: could use this section to generate an AST tree for each function, for use in GNNs if we want.
    # try:
    #     cleaned_tree = ast.parse(cleaned_code)
    #     serializable_cleaned_tree = ast.dump(cleaned_tree)
    # except SyntaxError:
    #     print('Syntax error in cleaned code')
    #     return None



    # write to file    
    # temp_input_dir = os.path.abspath("temp_input")

    # temp_filename = os.path.join(temp_input_dir, code['path'].split('/')[-1])
    
    # with open(temp_filename, "w") as tf:
    #     tf.write(cleaned_code)

        
    # temp_output_dir = os.path.abspath("temp_output")
    # output_file = os.path.join(temp_output_dir, code['path'].split('/')[-1])

    # cli_path = os.path.join('../', 'astminer', 'cli.sh')

    # if not os.path.isfile(cli_path):
    #     raise FileNotFoundError(f"The file {cli_path} was not found.")
    
    # original_dir = "../550Final-project/code/"
    # astminer_path = '../astminer' 
    # config_path = '../550Final-project/configs/astTree.yaml'

    # # Use astminer to create path contexts
    # call_astminer(original_dir, astminer_path, config_path)

    # # Do something to get the path contexts
    # path_contexts = None

    # # delete the file
    # os.remove(temp_filename)
    # os.remove(output_file)

    # what to do with the function labels: keep 1, keep all??

    return {
        # 'original_code': code_without_comments.strip(),
        'cleaned_code': cleaned_code,
        # 'original_tree': serializable_original_tree,
        # 'cleaned_tree': serializable_cleaned_tree,
        'description': comments.strip() if comments else None,
        # 'path_contexts': path_contexts
    }

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def call_astminer(original_dir, astminer_path, config_path):
    os.chdir(astminer_path)

    command = f"./cli.sh {config_path}"
    subprocess.run(command, shell=True, cwd=astminer_path)

    os.chdir(original_dir)


def process_chunk(chunk):
    processed_items = [process_dataset_item(code) for code in chunk['content']]
    return [item for item in processed_items if item is not None]

def create_dataset():
    dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python")
    num_processes = cpu_count()
    # num_processes = 1 # for testing with 1 core
    chunk_size = len(dataset['train']) // num_processes
    chunks = [dataset['train'][i:i + chunk_size] for i in range(0, len(dataset['train']), chunk_size)]

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)

    processed_data = [item for sublist in results for item in sublist]
    breakpoint()

    return processed_data

def create_dataset_for_testing():
    dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python")
    processed_data = []

    for code in dataset['train']:
        # print(code)
        processed_item = process_dataset_item(code)
        if processed_item:
            processed_data.append(processed_item)
    
    breakpoint()

    return processed_data

if __name__ == "__main__":
    testing = True
    if testing:
        create_dataset_for_testing()
    else:
        create_dataset()