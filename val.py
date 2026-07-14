from models.alg_model import AlgModel

if __name__ == '__main__':
    dataset_cfg = 'frag125_pku'
    device_no = '1'
    ckpt_path = r'ckpts/dpnet.pt'
    
    model = AlgModel(ckpt_path)

    metric = model.infer(
        data=f'./cfg/datasets/{dataset_cfg}.yaml',
        split='test',
        imgsz=384,
        device=device_no,
        save_json=True,
        name=f'runs/dpnet_preds'
    )
