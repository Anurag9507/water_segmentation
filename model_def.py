import segmentation_models_pytorch as smp

def get_model():
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,        # do NOT use imagenet here
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
    )
    return model