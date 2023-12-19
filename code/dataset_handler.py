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
import pandas as pd
from pathlib import Path
import shutil
import csv


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
    print(comments)
    code_without_comments = remove_chinese_characters(code_without_comments)

    if not comments and not code_without_comments:
        return None

    try:
        original_tree = ast.parse(code_without_comments)
    except SyntaxError:
        # print('Syntax error in original code')
        return None

    cleaned_code, clean_comments, name_mappings, function_sources = clean_code(code_without_comments)
    if function_sources is None:
        return []
    
    function_set = rename_functions(function_sources)

    path_contexts_set = []

    for function, func_name in function_set:
        temp_filename = os.path.join('/home/noah/COMP550/550Final-project/temp_input', code['path'].split('/')[-1])
        
        with open(temp_filename, "w") as tf:
            tf.write(function)

        temp_output_dir = os.path.abspath("temp_output")
        output_file = os.path.join(temp_output_dir, code['path'].split('/')[-1])
        

        cli_path = '/home/noah/COMP550/astminer/cli.sh'
        # cli_path = os.path.join(cli_path, '/cli.sh')

        # if not os.path.isfile(cli_path):
        #     raise FileNotFoundError(f"The file {cli_path} was not found."
        
        original_dir = "../550Final-project/code/"
        astminer_path = '/home/noah/COMP550/astminer/' 
        config_path = '../550Final-project/configs/astTree.yaml'

        # Use astminer to create path contexts
        call_astminer(original_dir, astminer_path, config_path)

        c2s_file_path = "/home/noah/COMP550/550Final-project/temp_output/py/data/path_contexts.c2s"

        # Do something to get the path contexts
        #cpath_contexts = read_path_contexts(c2s_file_path)

        token_mapping = load_mappings_to_dataframe('/home/noah/COMP550/550Final-project/temp_output/py/tokens.csv')
        node_type_mapping = load_mappings_to_dataframe('/home/noah/COMP550/550Final-project/temp_output/py/node_types.csv')
        path_mapping = load_mappings_to_dataframe('/home/noah/COMP550/550Final-project/temp_output/py/paths.csv')

        # processed_path_contexts = process_path_contexts(c2s_file_path, token_mapping, path_mapping)
        processed_data = read_and_process_c2s(c2s_file_path, token_mapping, path_mapping, node_type_mapping)
        if(processed_data == None):
            os.remove(temp_filename)
            shutil.rmtree('/home/noah/COMP550/550Final-project/temp_output/py')
            return

        # Save to a new file (optional)
        with open('/home/noah/COMP550/550Final-project/temp_output/processed_path_contexts.csv', 'a+', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(processed_data)
        breakpoint()

        # delete the file
        os.remove(temp_filename)
        shutil.rmtree('/home/noah/COMP550/550Final-project/temp_output/py')

        path_contexts_set.append((processed_data, func_name))

    return {
        # 'original_code': code_without_comments.strip(),
        'cleaned_code': cleaned_code,
        # 'original_tree': serializable_original_tree,
        'description': comments.strip(),
        'path_contexts': path_contexts_set
    }

def load_mappings_to_dataframe(file_path):
    return pd.read_csv(file_path, header=0, index_col=0).to_dict(orient='index')

def read_and_process_c2s(file_path, token_mapping, path_mapping, node_type_mapping):
    processed_data = []

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as file:
        for line in file:
            processed_line = process_path_context(line, token_mapping, path_mapping, node_type_mapping)
            processed_data.append(processed_line)

    return processed_data

def remove_chinese_characters(text):
    # Regular expression for matching Chinese characters
    chinese_char_pattern = r'[\u4e00-\u9fff]'

    # Replace Chinese characters with an empty string
    cleaned_text = re.sub(chinese_char_pattern, '', text)

    return cleaned_text

def process_path_context(line, token_mapping, path_mapping, node_type_mapping):
    label, *path_contexts = line.strip().split()
    processed_line = [label]

    for context in path_contexts:
        path_nodes = []
        parts = context.split(',')
        if len(parts) != 3:
            print(f"Invalid format in context: {context}")
            continue 

        start_token_id, path_id, end_token_id = parts

        path_nodes_info = path_mapping.get(int(path_id))
        if path_nodes_info is None:
            path_nodes = [{'Unknown': 'Unknown'}]
        else:
            if isinstance(path_nodes_info, dict):
                path_nodes_ids = []
                for value in path_nodes_info.values():
                    path_nodes_ids.extend(value.split())
            else:
                path_nodes_ids = path_nodes_info.split()

            for node_id in path_nodes_ids:
                node_type = node_type_mapping.get(int(node_id), 'Unknown')
                path_nodes.append({int(node_id): node_type['node_type']})

        path_nodes.insert(0,int(start_token_id))
        path_nodes.append(int(end_token_id))

        processed_line.append(path_nodes)

    return processed_line

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def call_astminer(original_dir, astminer_path, config_path):
    os.chdir(astminer_path)

    command = f"./cli.sh {config_path}"
    subprocess.run(command, shell=True, cwd=astminer_path)

    os.chdir(original_dir)


def process_chunk(chunk):
    processed_items = []
    for code in chunk['content']:
        path_contexts_set = process_dataset_item(code)
        if path_contexts_set:
            processed_items.extend(path_contexts_set)

    return [item for item in processed_items if item is not None]

def create_dataset():
    dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python")
    num_processes = cpu_count() - 2
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
        processed_item = process_dataset_item(code)
        if processed_item:
            processed_data.extend(processed_item)
    
    breakpoint()

    return processed_data

if __name__ == "__main__":
    testing = True
    if testing:
        create_dataset_for_testing()
    else:
        create_dataset()