from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import ast
from lib2to3 import refactor
import re


import ast
import builtins

class CodeCleaner(ast.NodeTransformer):
    def __init__(self):
        self.var_counter = 1
        self.func_counter = 1
        self.name_mapping = {}
        self.imported_modules = set()
        self.builtin_names = dir(builtins)

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
        if node.name not in self.builtin_names:
            original_name = node.name
            node.name = f'func{self.func_counter}'
            self.func_counter += 1
            self.name_mapping[original_name] = node.name
        self.generic_visit(node)
        return node


def convert_python2_to_python3(code):
    tool = refactor.RefactoringTool(refactor.get_fixers_from_package('lib2to3.fixes'))
    tree = tool.refactor_string(code, '<string>')
    return str(tree)

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
    return cleaned_code, comments, cleaner.name_mapping

def extract_and_remove_comments(code):
    comments = re.findall(r'(?m)^\s*#.*$', code)

    def replace_with_whitespace(match):
        return ' ' * len(match.group(0))

    code_without_comments = re.sub(r'(?m)^\s*#.*$', replace_with_whitespace, code)
    return ' '.join(comments), code_without_comments

def process_dataset_item(code):
    comments, code_without_comments = extract_and_remove_comments(code)
    if not comments and not code_without_comments:
        return None

    try:
        original_tree = ast.parse(code_without_comments)
        serializable_original_tree = ast.dump(original_tree)
    except SyntaxError:
        print('Syntax error in original code')
        return None

    cleaned_code, _, name_mapping = clean_code(code_without_comments)
    if cleaned_code is None:
        return None

    try:
        cleaned_tree = ast.parse(cleaned_code)
        serializable_cleaned_tree = ast.dump(cleaned_tree)
    except SyntaxError:
        print('Syntax error in cleaned code')
        return None

    return {
        'original_code': code_without_comments,
        'cleaned_code': cleaned_code,
        'original_tree': serializable_original_tree,
        'cleaned_tree': serializable_cleaned_tree,
        'description': comments
    }


def process_chunk(chunk):
    processed_items = [process_dataset_item(code) for code in chunk['content']]
    return [item for item in processed_items if item is not None]

def create_dataset():
    dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python")
    num_processes = cpu_count()
    # num_processes = 1 # for testing
    chunk_size = len(dataset['train']) // num_processes
    chunks = [dataset['train'][i:i + chunk_size] for i in range(0, len(dataset['train']), chunk_size)]

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)

    processed_data = [item for sublist in results for item in sublist]
    breakpoint()

    return processed_data

if __name__ == "__main__":
    create_dataset()