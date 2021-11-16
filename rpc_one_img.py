from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import json

# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'configs/rpc/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll.py'
checkpoint_file = 'result/faster_rcnn_r50_fpn_3x_rpc_protoS_ranking_mll/latest.pth'

# config_file1 = 'configs/rpc/cascade_rcnn_r50_fpn_3x_rpc.py'
# checkpoint_file1 = 'result/cascade_rcnn_r50_fpn_3x_rpc/latest.pth'
#
# config_file2 = 'configs/rpc/cascade_rcnn_r50_fpn_3x_rpc_proto2.py'
# checkpoint_file2 = 'result/cascade_rcnn_r50_fpn_3x_rpc_proto2/latest.pth'
#
# fp_idx1 = np.load('./result/cascade_rcnn_r50_fpn_3x_rpc/fp_idx.npy')
# fp_idx2 = np.load('./result/cascade_rcnn_r50_fpn_3x_rpc_proto2/fp_idx.npy')
# fp_idx3 = [x for x in fp_idx2 if x not in fp_idx1]
with open('/media/hao/Data/RPC/instances_test2019.json', 'r') as f:
    data = json.load(f)
    f.close()

# 根据配置文件和 checkpoint 文件构建模型
# model1 = init_detector(config_file1, checkpoint_file1, device='cuda:0')
# model2 = init_detector(config_file2, checkpoint_file2, device='cuda:0')
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# 测试单张图片并展示结果
img = '/media/hao/Data/RPC/test2019/20180824-15-44-39-474.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
img = mmcv.imread(img)
result = inference_detector(model, img)
# for i in range(len(fp_idx3)):
#     img = '/media/hao/Data/RPC/test2019/' + data['images'][fp_idx3[i]]['file_name']
#     img = mmcv.imread(img)
#     result1 = inference_detector(model1, img)
#     result2 = inference_detector(model2, img)
    # 在一个新的窗口中将结果可视化
# model.show_result(img, result, score_thr=0, show=True)
    # 或者将可视化结果保存为图片
    # idx = i
    # model1.show_result(img, result1, out_file='cascade_3x_fc/result{}_1.jpg'.format(idx), score_thr=0)
    # model2.show_result(img, result2, out_file='cascade_3x_proto3/result{}_2.jpg'.format(idx), score_thr=0)
model.show_result(img, result, out_file='faster_3x_rpc_protoS_ranking_mll/result.jpg', score_thr=0.35, thickness=4, font_size=13)
