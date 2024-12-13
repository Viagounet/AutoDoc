import ast
from typing import Callable, Optional

class ClassDescription:
    def __init__(self, name: str, signature: str, content: str, functions_called: list[Callable], inherits_from: Optional["ClassDescription"]):
        self.name = name
        self.signature = signature
        self.content = content
        self.functions_called = functions_called
        self.inherits_from = inherits_from

class FunctionDescription:
    def __init__(self, name: str, signature: str, content: str, functions_called: list[Callable]):
        self.name = name
        self.signature = signature
        self.content = content
        self.functions_called = functions_called

def extract_class_descriptions(file_path):
    """
    Extract detailed descriptions of classes from a Python file.

    Args:
        file_path (str): The path to the Python file.

    Returns:
        list[ClassDescription]: A list of ClassDescription objects describing the classes in the file.
    """
    class_descriptions = []
    class_map = {}

    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            tree = ast.parse(file_content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract class name
                name = node.name

                # Extract signature
                base_names = [base.id for base in node.bases if isinstance(base, ast.Name)]
                signature = f"class {name}({', '.join(base_names)})" if base_names else f"class {name}"

                # Extract content
                class_start = node.lineno - 1
                class_end = max(getattr(node, 'end_lineno', class_start), class_start)
                content = '\n'.join(file_content.splitlines()[class_start:class_end])

                # Extract functions called
                functions_called = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        functions_called.append(child.func.id)

                # Handle inheritance
                inherits_from = None
                if base_names:
                    for base_name in base_names:
                        if base_name in class_map:
                            inherits_from = class_map[base_name]
                            break

                class_description = ClassDescription(name, signature, content, functions_called, inherits_from)
                class_descriptions.append(class_description)
                class_map[name] = class_description

    except (FileNotFoundError, IOError) as e:
        print(f"Error reading file: {e}")
    except SyntaxError as e:
        print(f"Syntax error in file: {e}")

    return class_descriptions


def extract_function_descriptions(file_path):
    """
    Extract detailed descriptions of standalone functions from a Python file.

    Args:
        file_path (str): The path to the Python file.

    Returns:
        list[FunctionDescription]: A list of FunctionDescription objects describing the standalone functions in the file.
    """
    function_descriptions = []

    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            tree = ast.parse(file_content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if the function is standalone (not inside a class)
                parent = next((p for p in ast.walk(tree) if hasattr(p, 'body') and node in p.body), None)
                if not isinstance(parent, ast.ClassDef):
                    # Extract function name
                    name = node.name

                    # Extract signature
                    args = [arg.arg for arg in node.args.args]
                    signature = f"def {name}({', '.join(args)})"

                    # Extract content
                    func_start = node.lineno - 1
                    func_end = max(getattr(node, 'end_lineno', func_start), func_start)
                    content = '\n'.join(file_content.splitlines()[func_start:func_end])

                    # Extract functions called
                    functions_called = []
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                            functions_called.append(child.func.id)

                    function_description = FunctionDescription(name, signature, content, functions_called)
                    function_descriptions.append(function_description)

    except (FileNotFoundError, IOError) as e:
        print(f"Error reading file: {e}")
    except SyntaxError as e:
        print(f"Syntax error in file: {e}")

    return function_descriptions

classes = extract_function_descriptions("examples/summary.py")
for _class in classes:
    print(_class.signature)
