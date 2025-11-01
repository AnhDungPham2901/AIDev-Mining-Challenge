from tree_sitter import Language, Parser
import ast
# Load built-in Tree-sitter languages
from tree_sitter_languages import get_language

# Supported languages map
LANGUAGES = {
    "typescript": get_language("typescript"),
    "javascript": get_language("javascript"),
    "go": get_language("go"),
    "c_sharp": get_language("c_sharp"),
    "rust": get_language("rust"),
    "cpp": get_language("cpp"),
}

LANG_NODE_TYPES = {
    "cpp": {
        "function": ["function_definition"],
        "method": ["function_definition"],  # in class_specifier
        "class": ["class_specifier"],
    },
    "go": {
        "function": ["function_declaration"],
        "method": ["method_declaration"],
        "class": ["type_declaration"],
    },
    "rust": {
        "function": ["function_item"],
        "method": ["method_item"],
        "class": ["struct_item", "impl_item"],
    },
    "c_sharp": {
        "function": [],
        "method": ["method_declaration"],
        "class": ["class_declaration"],
    },
    "javascript": {
        "function": ["function_declaration", "arrow_function"],
        "method": ["method_definition"],
        "class": ["class_declaration"],
    },
    "typescript": {
        "function": ["function_declaration", "arrow_function"],
        "method": ["method_definition"],
        "class": ["class_declaration"],
    },
}

def parse_python(code: str):
    """Extract classes and functions using ast"""
    tree = ast.parse(code)
    classes, functions = [], []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Only add top-level functions, not methods inside classes
            if not isinstance(getattr(node, "parent", None), ast.ClassDef):
                functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        # Track parents
        for child in ast.iter_child_nodes(node):
            child.parent = node

    return {"classes": classes, "functions": functions}

def _is_inside_class(node, class_types):
    """Check if a node is inside a class"""
    parent = node.parent
    while parent:
        if parent.type in class_types:
            return True
        parent = parent.parent
    return False

def _extract_class_name(node):
    """Extract class name from a class node"""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode()
    return None

def _extract_function_name(node):
    """Extract function name from a regular function node"""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode()
    return None

def _extract_arrow_function_name(node):
    """Extract function name from an arrow function node"""
    if node.parent and node.parent.type == "variable_declarator":
        name_node = node.parent.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()
    return None

def parse_with_treesitter(lang_name, code: str):
    lang = LANGUAGES[lang_name]
    parser = Parser()
    parser.set_language(lang)
    tree = parser.parse(code.encode())
    root = tree.root_node

    classes, functions = [], []
    node_types = LANG_NODE_TYPES.get(lang_name, {})
    class_types = node_types.get("class", [])
    function_types = node_types.get("function", [])

    def walk(node):
        ntype = node.type
        
        # Handle class nodes
        if ntype in class_types:
            class_name = _extract_class_name(node)
            if class_name:
                classes.append(class_name)
        
        # Handle function nodes
        elif ntype in function_types:
            # Only add as function if it's NOT inside a class
            if not _is_inside_class(node, class_types):
                if ntype == "arrow_function":
                    func_name = _extract_arrow_function_name(node)
                else:
                    func_name = _extract_function_name(node)
                
                if func_name:
                    functions.append(func_name)

        for child in node.children:
            walk(child)

    walk(root)
    return {
        "classes": classes,
        "functions": functions,
    }

def extract_definitions(patch: str, lang: str = 'python'):
    import textwrap
    patch = textwrap.dedent(patch)
    if lang == "python":
        return parse_python(patch)
    elif lang in LANGUAGES:
        return parse_with_treesitter(lang, patch)
    else:
        return {"classes": [], "functions": []}

# Example usage
if __name__ == "__main__":
    # import textwrap

    # patch = textwrap.dedent("""
    #     def add(a, b):
    #         return a + b

    #     class Hello:
    #         def greet(self):
    #             print('hi')

    #     class World:
    #         def greet(self):
    #             print('world')

    #     class HelloWorld:
    #         def greet(self):
    #             print('hello world')
    # """)

    # patch = """#include <iostream>
    # using namespace std;

    # class Hello {
    # public:
    #     void greet() {
    #         cout << "hi" << endl;
    #     }
    # };

    # class World {
    # public:
    #     void greet() {
    #         cout << "world" << endl;
    #     }
    # };

    # int add(int a, int b) {
    #     return a + b;
    # }

    # int main() {
    #     Hello h;
    #     World w;
    #     h.greet();
    #     w.greet();
    #     cout << "Sum: " << add(2, 3) << endl;
    #     return 0;
    # }
    # """

    patch = """
    function add(a, b) {
    return a + b;
    }

    class Hello {
        greet() {
            console.log('hi');
        }
    }

    class World {
        greet() {
            console.log('world');
        }
    }

    const subtract = (a, b) => {
        return a - b;
    };
    """

    result = extract_definitions(patch, lang='javascript')
    print(result)
