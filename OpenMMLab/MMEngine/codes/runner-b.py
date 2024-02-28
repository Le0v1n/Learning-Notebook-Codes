from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam
from mmengine.runner import Runner


runner = Runner(
    # 你的模型
    model=MyAwesomeModel(
        layers=2,
        activation='relu'),
    # 模型检查点、日志等都将存储在工作路径中
    work_dir='exp/my_awesome_model',

    # 训练所用数据
    train_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=True,
            size=10000),
        shuffle=True,
        collate_fn=default_collate,
        batch_size=64,
        pin_memory=True,
        num_workers=2),
    # 训练相关配置
    train_cfg=dict(
        by_epoch=True,   # 根据 epoch 计数而非 iteration
        max_epochs=10,
        val_begin=2,     # 从第 2 个 epoch 开始验证
        val_interval=1), # 每隔 1 个 epoch 进行一次验证

    # 优化器封装，MMEngine 中的新概念，提供更丰富的优化选择。
    # 通常使用默认即可，可缺省。有特殊需求可查阅文档更换，如
    # 'AmpOptimWrapper' 开启混合精度训练
    optim_wrapper=dict(
        optimizer=dict(
            type=Adam,
            lr=0.001)),
    # 参数调度器，用于在训练中调整学习率/动量等参数
    param_scheduler=dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1),

    # 验证所用数据
    val_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=False,
            size=1000),
        shuffle=False,
        collate_fn=default_collate,
        batch_size=1000,
        pin_memory=True,
        num_workers=2),
    # 验证相关配置，通常为空即可
    val_cfg=dict(),
    # 验证指标与验证器封装，可自由实现与配置
    val_evaluator=dict(type=Accuracy),

    # 以下为其他进阶配置，无特殊需要时尽量缺省
    # 钩子属于进阶用法，如无特殊需要，尽量缺省
    default_hooks=dict(
        # 最常用的默认钩子，可修改保存 checkpoint 的间隔
        checkpoint=dict(type='CheckpointHook', interval=1)),

    # `luancher` 与 `env_cfg` 共同构成分布式训练环境配置
    launcher='none',
    env_cfg=dict(
        cudnn_benchmark=False,   # 是否使用 cudnn_benchmark
        backend='nccl',   # 分布式通信后端
        mp_cfg=dict(mp_start_method='fork')),  # 多进程设置
    log_level='INFO',

    # 加载权重的路径 (None 表示不加载)
    load_from=None,
    # 从加载的权重文件中恢复训练
    resume=False
)

# 开始训练你的模型吧
runner.train()