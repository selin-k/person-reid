from .resnet import *

__model_factory = {
    'resnet50': resnet50
}

def get_model_names():
    """Displays available models
    """
    print(list(__model_factory.keys()))



def get_model(name, num_classes, pretrained=True, **kwargs):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )