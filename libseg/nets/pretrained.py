import segmentation_models_pytorch as smp

def get_deeplabplus(encoder_name = 'resnet34'):
    net = smp.DeepLabV3Plus(
    encoder_name=encoder_name,
    encoder_depth = 5,
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                     # model output channels (number of classes in your dataset)
    )

    return net


def get_deeplab(encoder_name = 'resnet34'):
    net = smp.DeepLabV3(
        encoder_name=encoder_name,
        encoder_depth=5,
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    return net