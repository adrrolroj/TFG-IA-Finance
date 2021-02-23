import torchvision.models as models


def get_list_models():
    pretrained_models = []
    pretrained_models.append(('vgg16', models.vgg16(pretrained=True)))
    pretrained_models.append(('alexnet', models.alexnet(pretrained=True)))
    #pretrained_models.append(('resnet18', models.resnet18(pretrained=True)))
    pretrained_models.append(('squeezenet', models.squeezenet1_0(pretrained=True)))
    #pretrained_models.append(('densenet', models.densenet161(pretrained=True)))
    pretrained_models.append(('googlenet', models.googlenet(pretrained=True)))
    pretrained_models.append(('shufflenet', models.shufflenet_v2_x1_0(pretrained=True)))
    pretrained_models.append(('mobilenet', models.mobilenet_v2(pretrained=True)))
    #pretrained_models.append(('resnext50', models.resnext50_32x4d(pretrained=True)))
    pretrained_models.append(('wide_resnet', models.wide_resnet50_2(pretrained=True)))
    #pretrained_models.append(('mnasnet', models.mnasnet1_0(pretrained=True)))
    #pretrained_models.append(('inception', models.inception_v3(pretrained=True)))
    return pretrained_models
