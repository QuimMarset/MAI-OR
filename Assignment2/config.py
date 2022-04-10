_base_ = [
    'D:/MAI/OR/Lab/MAI-OR/Assignment2/mmsegmentation/configs/_base_/models/upernet_swin.py'
]


num_classes = 47

model = dict(
    pretrained='D:/MAI/OR/Lab/MAI-OR/Assignment2/mmsegmentation/checkpoints/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.pth',
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer_config = dict()
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=1000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')


log_config = dict(
    interval=500,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
gpu_ids = range(1)


data_root = 'D:/MAI/OR/Lab/MAI-OR/Assignment2/fashionpedia/'
dataset_type = 'Fashionpedia'

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True)
img_scale = (224, 224)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(1.0, 1.0)),
    dict(type='RandomFlip', prob=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip', prob=0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

work_dir = 'D:/MAI/OR/Lab/MAI-OR/Assignment2/results/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train', 
        ann_dir='segmentations/train',
        pipeline=train_pipeline,
        split=None),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='segmentations/val',
        pipeline=val_pipeline,
        split='D:/MAI/OR/Lab/MAI-OR/Assignment2/val_split.txt'),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='segmentations/val',
        pipeline=val_pipeline,
        split='D:/MAI/OR/Lab/MAI-OR/Assignment2/val_split.txt')
)