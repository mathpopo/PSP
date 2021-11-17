from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import json

config_file = 'configs/rpc/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll.py'
checkpoint_file = 'result/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll/latest.pth'

with open('/media/xxx/Data/RPC/instances_test2019.json', 'r') as f:
    data = json.load(f)
    f.close()


model = init_detector(config_file, checkpoint_file, device='cuda:0')


img = '/media/xxx/Data/RPC/test2019/20180824-15-44-39-474.jpg'  
img = mmcv.imread(img)
result = inference_detector(model, img)

model.show_result(img, result, out_file='faster_3x_rpc_protoS_srr_mll/result.jpg')
