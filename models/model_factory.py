import torch.nn as nn
from models.rangeplace import featureExtracter
from models.overlap_transformer import OT


from models.minkloc import MinkLoc
from misc.utils import ModelParams
from models.layers.eca_block import ECABasicBlock
from models.minkfpn import MinkFPN
from models.layers.pooling_wrapper import PoolingWrapper


def model_factory(model_params: ModelParams):
    in_channels = 1

    if model_params.model == 'RangePlace':
        print('Model: {}'.format(model_params.model))
        model = featureExtracter()
    elif model_params.model == 'MinkLoc':
        print('Model: {}'.format(model_params.model))
        block_module = create_resnet_block(model_params.block)
        backbone = MinkFPN(in_channels=in_channels, out_channels=model_params.feature_size,
                           num_top_down=model_params.num_top_down, conv0_kernel_size=model_params.conv0_kernel_size,
                           block=block_module, layers=model_params.layers, planes=model_params.planes)
        pooling = PoolingWrapper(pool_method=model_params.pooling, in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)
        model = MinkLoc(backbone=backbone, pooling=pooling, normalize_embeddings=model_params.normalize_embeddings)
        
    elif model_params.model == 'OT':
        print('Model: {}'.format(model_params.model))
        model = OT()
    elif model_params.model == 'ppt-net':
        from models.ppt.pptnet_applier import get_ppt_net_model
        print('Model: {}'.format(model_params.model))
        model = get_ppt_net_model()
    else:
        raise NotImplementedError('Model not implemented: {}'.format(model_params.model))

    return model


def create_resnet_block(block_name: str) -> nn.Module:
    if block_name == 'BasicBlock':
        block_module = BasicBlock
    elif block_name == 'Bottleneck':
        block_module = Bottleneck
    elif block_name == 'ECABasicBlock':
        block_module = ECABasicBlock
    else:
        raise NotImplementedError('Unsupported network block: {}'.format(block_name))

    return block_module
