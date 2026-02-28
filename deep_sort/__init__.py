from .deep_sort import DeepSort

__all__ = ['DeepSort', 'build_tracker']


def build_tracker(cfg, use_cuda):
    if cfg.USE_FASTREID:
        return DeepSort(model_path=cfg.FASTREID.CHECKPOINT, model_config=cfg.FASTREID.CFG,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=use_cuda)

    else:
        # 核心修改：硬指定resnet18权重路径，替换原cfg.DEEPSORT.REID_CKPT
        return DeepSort(model_path="./deep_sort/deep/checkpoint/resnet18-5c106cde.pth",
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=use_cuda)