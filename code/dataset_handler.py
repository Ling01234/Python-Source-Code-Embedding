from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import ast
from lib2to3 import refactor
import re
import ast
import builtins

class PathConstructor:
    def __init__(self):
        self.terminal_nodes = []
        self.path_contexts = []

    def is_terminal(self, node):
        return not any(isinstance(child, ast.AST) for child in ast.iter_child_nodes(node))

    def find_terminal_nodes(self, node):
        if self.is_terminal(node):
            if not isinstance(node, ast.alias):
                self.terminal_nodes.append(node)
        else:
            for child in ast.iter_child_nodes(node):
                self.find_terminal_nodes(child)

    def get_parent_map(self, node, parent=None, parent_map=None):
        if parent_map is None:
            parent_map = {}
        parent_map[node] = parent
        for child in ast.iter_child_nodes(node):
            self.get_parent_map(child, node, parent_map)
        return parent_map

    def find_lowest_common_ancestor(self, node1, node2, parent_map):
        ancestors = set()
        while node1:
            ancestors.add(node1)
            node1 = parent_map[node1]
        while node2 not in ancestors:
            node2 = parent_map[node2]
        return node2

    def construct_path(self, start, end, parent_map):
        lca = self.find_lowest_common_ancestor(start, end, parent_map)

        path = []
        # Trace path from start to LCA
        node = start
        while node != lca:
            path.append(self.get_node_label(node) + '↑')
            node = parent_map[node]
        
        # Trace path from LCA to end
        node = end
        while node != lca:
            path.append(self.get_node_label(node) + '↓')
            node = parent_map[node]

        path_str = ' '.join(path)
        return path_str


    def generate_path_contexts(self, ast_tree):
        parent_map = self.get_parent_map(ast_tree)
        self.find_terminal_nodes(ast_tree)
        for i, start in enumerate(self.terminal_nodes):
            for end in self.terminal_nodes[i + 1:]:
                path = self.construct_path(start, end, parent_map)
                if path != '':
                    self.path_contexts.append((self.get_node_label(start), path, self.get_node_label(end)))
        return self.path_contexts

    # def get_node_label(self, node):
    #     # breakpoint()
    #     if isinstance(node, ast.Name):
    #         return node.id  # Variable name
    #     elif isinstance(node, ast.Expr):
    #         return node.id
    #     elif isinstance(node, ast.Constant):
    #         return str(node.value)  # Constant value
    #     elif isinstance(node, ast.FunctionDef):
    #         return node.name  # Function name
    #     elif isinstance(node, ast.arg):
    #         return node.arg  # Function argument name
    #     else:
    #         return type(node).name  # Fallback to node type

    def get_node_label(self, node):
        # Check if the node is a terminal node and return the value
        if isinstance(node, (ast.Name, ast.Constant, ast.FunctionDef, ast.arg, ast.Expr)):
            if hasattr(node, 'id'):
                return node.id
            elif hasattr(node, 'value'):
                return str(node.value)
            elif hasattr(node, 'name'):
                return node.name
            elif hasattr(node, 'arg'):
                return node.arg
            else:
                # This will handle ast.Expr and other terminal nodes
                return type(node).__name__
        else:
            return type(node).__name__  # Fallback to node type


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

    # # testing out path creating system
    # path_constructor = PathConstructor()
    # paths = path_constructor.generate_path_contexts(tree)

    # breakpoint()
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
        'original_code': code_without_comments.strip(),
        'cleaned_code': cleaned_code.strip(),
        'original_tree': serializable_original_tree,
        'cleaned_tree': serializable_cleaned_tree,
        'description': comments.strip()
    }


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
        processed_item = process_dataset_item(code['content'])
        if processed_item:
            processed_data.append(processed_item)
    
    breakpoint()

    return processed_data

if __name__ == "__main__":
    testing = False
    if testing:
        create_dataset_for_testing()
    else:
        create_dataset()