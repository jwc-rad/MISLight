import importlib
import inspect
import os
import pkgutil

def recursive_find_python_class(class_name, current_module):
    tr = None
    m = importlib.import_module(current_module)
    folder = [os.path.dirname(inspect.getfile(m))]
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(class_name, next_current_module)
            if tr is not None:
                break
    return tr