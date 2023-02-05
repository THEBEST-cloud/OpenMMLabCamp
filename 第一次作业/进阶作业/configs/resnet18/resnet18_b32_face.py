_base_ = ['../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py', '../_base_/default_runtime.py']


model = dict(
    head=dict(
    num_classes=7,
    topk = (1,)
    ))

data = dict(
    samples_per_gpu = 32,
    workers_per_gpu = 2,
    train = dict(
    data_prefix = 'data/Face/train',
    ann_file = 'data/Face/train.txt',
    classes = 'data/Face/classes.txt'
    ),
    val = dict(
    data_prefix = 'data/Face/val',
    ann_file = 'data/Face/val.txt',
    classes = 'data/Face/classes.txt'
    )
    )

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    step=[1])

runner = dict(type='EpochBasedRunner', max_epochs=400)
checkpoint_config = dict(interval=5)
# 预训练模型
load_from='/HOME/scz0aui/run/mmclassfication/work_dirs/resnet18_b32_face/epoch_100.pth'
