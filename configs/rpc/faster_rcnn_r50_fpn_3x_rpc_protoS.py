# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(proto=1,  # 1:S, 2:C
                       ranking=False,
                       num_classes=200)),
    test_cfg=dict(
        rcnn=dict(
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
            max_per_img=20)
    )
)

# 修改数据集相关设置
dataset_type = 'RpcDataset'
data_root = '/media/hao/Data/RPC/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        type=dataset_type,
        img_prefix='/media/hao/Data/RPC/val2019/',
        ann_file='/media/hao/Data/RPC/instances_val2019.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix='/media/hao/Data/RPC/val2019/',
        ann_file='/media/hao/Data/RPC/instances_val2019.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix='/media/hao/Data/RPC/test2019/',
        ann_file='/media/hao/Data/RPC/instances_test2019.json',
        pipeline=test_pipeline)
)

# 我们可以使用预训练的 Faster R-CNN 来获取更好的性能
load_from = 'checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

work_dir = './result/faster_rcnn_r50_fpn_3x_rpc_protoS'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
# optimizer
optimizer = dict(type='SGD', lr=0.02 / 8, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8 * 3, 11 * 3])
runner = dict(type='EpochBasedRunner', max_epochs=12 * 3)
log_config = dict(interval=100)
checkpoint_config = dict(interval=12)
evaluation = dict(interval=12 * 3)
