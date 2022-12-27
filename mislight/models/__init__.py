from mislight.utils.find_class import recursive_find_python_class

def create_model(opt):
    model = recursive_find_python_class(opt.model, 'mislight.models')
    instance = model(opt)
    print(f'model [{type(instance).__name__}] was created')
    return instance