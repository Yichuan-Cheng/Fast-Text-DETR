- Fork from DPText-DETR

- Set up environment as :

  - ```shell
    conda create -n DPText-DETR python=3.8 -y
    conda activate DPText-DETR
    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install opencv-python scipy timm shapely albumentations Polygon3
    python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
    pip install setuptools==59.5.0
    git clone https://github.com/ymy-k/DPText-DETR.git
    cd DPText-DETR
    python setup.py build develop
    ```

​	Use our repository to replace its files

- run command:

  - ```shell
    python tools/train_net.py --config-file configs/DPText_DETR/CTW1500/R_50_poly.yaml --num-gpus 2
    ```