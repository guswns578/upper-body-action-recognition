"""Inference-only adapter for official PYSKL ST-GCN++ (no new architecture).

Executes upstream class/function definitions unchanged, excluding registry decorators
and package imports that pull in unrelated CUDA/video/training dependencies.
Selected upstream sources and their license are preserved under vendor/pyskl.
The loss builder is unused during inference; training is deliberately unsupported.
"""
import ast
import copy as cp
from abc import ABCMeta, abstractmethod
from pathlib import Path
import numpy as np
import torch
from torch import nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import normal_init

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'vendor/pyskl'

def _load(relative, namespace, names=None):
    path=SOURCE/relative
    tree=ast.parse(path.read_text(encoding='utf-8'))
    definitions=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and (names is None or node.name in names):
            node.decorator_list=[]  # only module-level registry decorators
            definitions.append(node)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),'exec'),namespace)

def official_components():
    ns=dict(np=np,torch=torch,nn=nn,cp=cp,EPS=1e-4,ABCMeta=ABCMeta,abstractmethod=abstractmethod,
            build_activation_layer=build_activation_layer,build_norm_layer=build_norm_layer,
            normal_init=normal_init,build_loss=lambda cfg:nn.CrossEntropyLoss())
    _load('utils/graph.py',ns)
    _load('models/gcns/utils/init_func.py',ns)
    _load('models/gcns/utils/gcn.py',ns,{'unit_gcn'})
    _load('models/gcns/utils/tcn.py',ns,{'unit_tcn','mstcn'})
    _load('models/gcns/stgcn.py',ns,{'STGCNBlock','STGCN'})
    _load('models/heads/base.py',ns,{'BaseHead'})
    _load('models/heads/simple_head.py',ns,{'SimpleHead','GCNHead'})
    _load('datasets/pipelines/pose_related.py',ns,{'PreNormalize2D','PoseDecode','FormatGCNInput'})
    _load('datasets/pipelines/sampling.py',ns,{'UniformSampleFrames','UniformSample'})
    return ns

class ActionModel(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        self.official=official_components()
        self.backbone=self.official['STGCN'](graph_cfg=dict(layout='coco',mode='spatial'),
            gcn_adaptive='init',gcn_with_res=True,tcn_type='mstcn')
        self.cls_head=self.official['GCNHead'](num_classes=120,in_channels=256)
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.load_state_dict(weights.get('state_dict',weights),strict=True)
        self.eval()

    def preprocess(self,xy,scores,shape,num_clips=1):
        """Input [persons,frames,17,2], [persons,frames,17]. Official joint stream.

        GenSkeFeat(feats=['j']) only renames/merges this single tensor; no numerical
        transform is required after official PreNormalize2D appends scores.
        """
        assert xy.shape[-2:]==(17,2) and scores.shape==xy.shape[:-1]
        assert xy.shape[1]>0  # Upstream loops short offline clips; streaming runner requires 100 real frames.
        result=dict(keypoint=xy.copy(),keypoint_score=scores.copy(),img_shape=shape,
                    total_frames=xy.shape[1],start_index=0,test_mode=True)
        for transform in [self.official['PreNormalize2D'](),
                          self.official['UniformSample'](clip_len=100,num_clips=num_clips),
                          self.official['PoseDecode'](),self.official['FormatGCNInput'](num_person=2)]:
            result=transform(result)
        return torch.from_numpy(result['keypoint']),result['frame_inds'].tolist()

    @torch.inference_mode()
    def predict(self,xy,scores,shape,num_clips=1):
        data,indices=self.preprocess(xy,scores,shape,num_clips)
        # Official RecognizerGCN: per-clip softmax, then average probabilities.
        logits=self.cls_head(self.backbone(data))
        return logits.softmax(-1).mean(0).numpy(),indices
