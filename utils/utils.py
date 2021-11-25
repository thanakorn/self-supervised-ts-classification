def get_class(module_name, cls_name):
    module = __import__(module_name, fromlist=[cls_name])
    return getattr(module, cls_name)

def get_optimizer(optimizer: str):
    return get_class('torch.optim', optimizer)

def get_loss_fn(loss_fn: str):
    return get_class('torch.nn', loss_fn)