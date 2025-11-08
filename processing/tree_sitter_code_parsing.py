# ============================================================================
# TREE-SITTER BASED FUNCTION/CLASS EXTRACTION FROM PATCH DIFFS
# ============================================================================

## Dont use this in the project because there are some issues to improve: 1. patch diff is not full code. 
import os
import pandas as pd
from loguru import logger
from tree_sitter import Language, Parser
# Load built-in Tree-sitter languages
from tree_sitter_languages import get_language

# Supported languages map
LANGUAGES = {
    "python": get_language("python"),
    "typescript": get_language("typescript"),
    "javascript": get_language("javascript"),
    "go": get_language("go"),
    "c_sharp": get_language("c_sharp"),
    "java": get_language("java"),
    "cpp": get_language("cpp"),
}

EXT_TO_LANG = {
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",

    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",

    ".ts": "typescript",
    ".tsx": "typescript",
    
    ".go": "go",
}


def detect_lang_from_filename(filename: str):
    """Detect programming language from filename extension."""
    _, ext = os.path.splitext(filename.lower())
    return EXT_TO_LANG.get(ext)

def parse_patch_to_code_versions(patch: str):
    """
    Parse a unified diff patch to extract old and new code versions.
    
    Args:
        patch: Unified diff string
        
    Returns:
        tuple: (old_code, new_code) as strings, or (None, None) if parsing fails
    """
    if not patch:
        return None, None
    
    old_lines = []
    new_lines = []
    
    for line in patch.split('\n'):
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            continue
        elif line.startswith('-'):
            # Line removed from old version
            old_lines.append(line[1:])
        elif line.startswith('+'):
            # Line added to new version
            new_lines.append(line[1:])
        else:
            # Context line (present in both)
            old_lines.append(line[1:] if line.startswith(' ') else line)
            new_lines.append(line[1:] if line.startswith(' ') else line)
    
    old_code = '\n'.join(old_lines)
    new_code = '\n'.join(new_lines)
    
    return old_code, new_code


def extract_symbols_with_treesitter(code: str, language: str):
    """
    Extract function and class definitions from code using Tree-sitter.
    
    Args:
        code: Source code string
        language: Language name (must be in LANGUAGES dict)
        
    Returns:
        dict: {'functions': set(), 'classes': set()}
    """
    symbols = {'functions': set(), 'classes': set()}
    
    if not code or not code.strip():
        return symbols
    
    # Map language names to Tree-sitter language objects
    lang_mapping = {
        'python': 'python',
        'javascript': 'javascript',
        'java': 'java',
        'cpp': 'cpp',
        'c-sharp': 'c_sharp',
        'go': 'go',
        'typescript': 'typescript',
    }
    
    ts_lang_name = lang_mapping.get(language)
    if not ts_lang_name or ts_lang_name not in LANGUAGES:
        logger.warning(f"Language {language} not supported by program")
        return symbols
    
    try:
        # Create parser
        parser = Parser()
        parser.set_language(LANGUAGES[ts_lang_name])
        
        # Parse the code
        tree = parser.parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        
        # Define queries for different languages
        if ts_lang_name == 'python':
            function_types = ['function_definition']
            class_types = ['class_definition']
            name_field = 'name'
            
        elif ts_lang_name == 'javascript':
            function_types = ['function_declaration', 'function_expression', 
                            'arrow_function', 'method_definition']
            class_types = ['class_declaration']
            name_field = 'name'
            
        elif ts_lang_name == 'java':
            function_types = ['method_declaration', 'constructor_declaration']
            class_types = ['class_declaration', 'interface_declaration']
            name_field = 'name'
            
        elif ts_lang_name == 'cpp':
            function_types = ['function_definition', 'function_declarator']
            class_types = ['class_specifier', 'struct_specifier']
            name_field = 'declarator'
            
        elif ts_lang_name == 'c_sharp':
            function_types = ['method_declaration', 'constructor_declaration']
            class_types = ['class_declaration', 'interface_declaration']
            name_field = 'name'
        elif ts_lang_name == 'go':
            function_types = ['function_declaration']
            class_types = ['type_declaration']
            name_field = 'name'
        elif ts_lang_name == 'typescript':
            function_types = ['function_declaration', 'function_expression', 'arrow_function', 'method_definition']
            class_types = ['class_declaration']
            name_field = 'name'
        else:
            return symbols
        
        # Traverse tree to find function and class definitions
        def traverse(node):
            if node.type in function_types:
                name_node = node.child_by_field_name(name_field)
                if name_node:
                    symbols['functions'].add(name_node.text.decode('utf8'))
            
            if node.type in class_types:
                name_node = node.child_by_field_name(name_field)
                if name_node:
                    symbols['classes'].add(name_node.text.decode('utf8'))
            
            # Recurse through children
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        
    except Exception as e:
        raise e
    
    return symbols


def extract_changes_from_patch(patch: str, language: str) -> dict:
    """
    Extract added and removed functions/classes from a patch diff.
    
    IMPLEMENTATION:
    Step 1: Split diff into old_code and new_code
    Step 2: Use Tree-sitter parsing for structural extraction
    Step 3: Compare symbol sets to identify added/removed
    
    Args:
        patch: Unified diff string
        language: Programming language (must match EXT_TO_LANG values)
        
    Returns:
        dict: {
            'added_functions': set(),
            'removed_functions': set(),
            'added_classes': set(),
            'removed_classes': set()
        }
    """
    result = {
        'added_functions': set(),
        'removed_functions': set(),
        'added_classes': set(),
        'removed_classes': set()
    }
    
    if not patch or not language:
        return result
    
    # STEP 1: Split diff into old_code and new_code
    old_code, new_code = parse_patch_to_code_versions(patch)
    print("This is old_code", old_code)
    print("This is new_code", new_code)
    
    if old_code is None or new_code is None:
        return result
    
    # STEP 2: Try Tree-sitter parsing for structural extraction
    try:
        old_symbols = extract_symbols_with_treesitter(old_code, language)
    except Exception as e:
        raise e
    try:
        new_symbols = extract_symbols_with_treesitter(new_code, language)
    except Exception as e:
        raise e
    
    # STEP 3: Compare symbol sets to identify added/removed
    result['added_functions'] = new_symbols['functions'] - old_symbols['functions']
    result['removed_functions'] = old_symbols['functions'] - new_symbols['functions']
    result['added_classes'] = new_symbols['classes'] - old_symbols['classes']
    result['removed_classes'] = old_symbols['classes'] - new_symbols['classes']
    
    return result


def analyze_patch_changes(patch: str, filename: str):
    """
    High-level function to analyze a patch and extract all changes.
    
    Args:
        patch: Unified diff string
        filename: Filename to detect language
        
    Returns:
        dict: Change analysis results
    """
    # Detect language from filename
    language = detect_lang_from_filename(filename)
    
    if not language:
        raise ValueError(f"Language not supported: {filename}")
    
    try:
        changes = extract_changes_from_patch(patch, language)
        changes['language'] = language
    except Exception as e:
        logger.error(f"Error analyzing patch for {filename}: {e}")
        return {
            'language': None,
            'added_functions': set(),
            'removed_functions': set(),
            'added_classes': set(),
            'removed_classes': set(),
            'error': str(e)
        }   
    return changes


def extract_changes_for_df(row):
    """
    Extract function/class changes from a DataFrame row.
    Designed to be used with pandas DataFrame.apply().
    
    Args:
        row: pandas Series with 'patch' and 'filename' columns
        
    Returns:
        pandas Series with 4 columns containing sets:
        - added_functions: set of function names added
        - added_classes: set of class names added  
        - removed_functions: set of function names removed
        - removed_classes: set of class names removed
        
    Usage:
        df[['added_functions', 'added_classes', 'removed_functions', 'removed_classes']] = \\
            df.apply(extract_changes_for_df, axis=1)
            
        # Or with progress bar:
        from tqdm import tqdm
        tqdm.pandas()
        df[['added_functions', 'added_classes', 'removed_functions', 'removed_classes']] = \\
            df.progress_apply(extract_changes_for_df, axis=1)
    """
    # Define column names and order
    cols = ['added_functions', 'added_classes', 'removed_functions', 'removed_classes']
    
    # Handle missing values
    if pd.isna(row['patch']) or pd.isna(row['filename']):
        return pd.Series([set(), set(), set(), set()], index=cols, dtype=object)
    
    try:
        # Detect language from filename
        language = detect_lang_from_filename(row['filename'])
        
        if not language:
            return pd.Series([set(), set(), set(), set()], index=cols, dtype=object)
        
        # Extract changes
        changes = extract_changes_from_patch(row['patch'], language)
        
        # Return as Series with explicit index and dtype
        return pd.Series([
            changes['added_functions'],
            changes['added_classes'],
            changes['removed_functions'],
            changes['removed_classes']
        ], index=cols, dtype=object)
        
    except Exception as e:
        # Log error but don't break the pipeline
        logger.debug(f"Error processing {row.get('filename', 'unknown')}: {e}")
        return pd.Series([set(), set(), set(), set()], index=cols, dtype=object)
