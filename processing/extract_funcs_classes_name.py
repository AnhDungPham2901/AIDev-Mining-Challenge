# given a patch extract the functions names
# the pattern to extract: always get lines that start with + def
# return a list of function names, or None if empty
# return only name, not include the parentheses

def extract_functions(patch: str) -> list | None:
    # split the patch into lines
    lines = patch.splitlines()
    # extract the function names
    funcs = []
    for line in lines:
        if line.startswith('+def'):
            func_name = line.split('def')[1].strip().split('(')[0].strip()
            if func_name:
                funcs.append(func_name)
    return funcs if funcs else None


# given a patch extract the classes names
# the pattern to extract: always get lines that start with + class
# return a list of class names, or None if empty
# return only name, not include the parentheses

def extract_classes(patch: str) -> list | None:
    # split the patch into lines
    lines = patch.splitlines()
    # extract the class names
    classes = []
    for line in lines:
        if line.startswith('+class'):
            class_name = line.split('class')[1].strip().split(':')[0].strip()
            if class_name:
                classes.append(class_name)
    return classes if classes else None


# extract functions are removed
# the pattern to extract: always get lines that start with -def
# return a list of function names, or None if empty
# return only name, not include the parentheses

def extract_removed_functions(patch: str) -> list | None:
    # split the patch into lines
    lines = patch.splitlines()
    # extract the function names
    funcs = []
    for line in lines:
        if line.startswith('-def'):
            func_name = line.split('def')[1].strip().split('(')[0].strip()
            if func_name:
                funcs.append(func_name)
    return funcs if funcs else None


# extract classes are removed
# the pattern to extract: always get lines that start with -class
# return a list of class names, or None if empty
# return only name, not include the parentheses

def extract_removed_classes(patch: str) -> list | None:
    # split the patch into lines
    lines = patch.splitlines()
    # extract the class names
    classes = []
    for line in lines:
        if line.startswith('-class'):
            class_name = line.split('class')[1].strip().split(':')[0].strip()
            if class_name:
                classes.append(class_name)
    return classes if classes else None