import torch
import models_vit


def mae(model_size='base', apply_pool=False, global_pool=False, num_classes=0, pretrained=True, **kwargs):
    """
    ViT-Base/16x16 pre-trained with MAE.
    Achieves 83.6% top-1 accuracy on ImageNet after finetuning.
    """

    # Name
    model_name = f"vit_{model_size}_patch16"
    assert model_name in ['vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch16'], f'{model_name} invalid'
    
    # Create model
    model = models_vit.__dict__[model_name](num_classes=num_classes, apply_pool=apply_pool, global_pool=global_pool)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=f"https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_{model_size}.pth",
            map_location="cpu",
        )['model']
        model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == "__main__":
    model = mae()
    import pdb
    pdb.set_trace()
