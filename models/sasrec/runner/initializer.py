from models.sasrec.modules.sasrec import SASRec
from models.sasrec.modules.nova import Novafuser
from models.sasrec.modules.l2rec import L2Rec
from models.sasrec.modules.srgnn import SRGNN
from models.sasrec.modules.namerec import NameRec

model_mapping = {}


def register_model(name):
    def _thunk(func):
        if model_mapping.get(name, -1) == -1:
            model_mapping[name] = func
        return func

    return _thunk


def build_model(name):
    if callable(name):
        return name
    elif name in model_mapping:
        return model_mapping[name]
    else:
        raise ValueError("Unknown model name : {}".format(name))


@register_model("sasrec")
def sasrec(config, num_user, num_item, device, **kwargs):
    return SASRec(config, num_user, num_item, device)


@register_model("nova")
def nova(config, num_user, num_item, device, **kwargs):
    return Novafuser(config, num_user, num_item, device, num_meta=kwargs['num_meta'], meta_information=kwargs['meta_information'])


@register_model("l2rec")
def l2rec(config, num_user, num_item, device, **kwargs):
    return L2Rec(config, num_user, num_item, device)


@register_model("namerec")
def namerec(config, num_user, num_item, device, **kwargs):
    return NameRec(config, num_user, num_item, device, productname_table=kwargs['productname_table'])


@register_model("srgnn")
def srgnn(config, num_user, num_item, device, **kwargs):
    return SRGNN(config, num_user, num_item, device)
