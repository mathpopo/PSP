# PSP: Automatic Check-out via Prototype-based Classifier Learning from Single-Product Exemplars
--------------------------
## Environment

Python 3.7.11  
Pytorch 1.7.1  
torchvision 0.8.2  
cudatoolkit 10.1.243  
mmcv-full 1.3.9  
mmdet 2.14.0

--------------------------
## Installation

The installation process of our code can follow the MMDetection in the url: https://mmdetection.readthedocs.io/en/v2.14.0/get_started.html 
and overlay the original files by our code.  
Or you can download our code and the file "README.md" of MMDetection, then run ''python setup.py develop'' .  

--------------------------
## Dataset

You can download the RPC dataset by this two urls:   
    https://www.kaggle.com/diyer22/retail-product-checkout-dataset  
    https://pan.baidu.com/s/1vrrLaSpJe5JxT3zhYfOaog  
    
--------------------------
## Necessary modifications

Before training or testing, you must modify the contents of the paths in the following files.

1. configs/rpc/xxx.py  
    In these configuration files, you need modify the following parameters.  
    
    data_root = '{your_dataset_root_path}'  
    img_prefix = '{your_train_or_test_dataset_path}'  
    ann_file = '{your_train_or_test_annotation_file}'  
    
2. rpc_eval.py  
    The default value of '--ann_file' can be modified as '{your_train_or_test_annotation_file}'.  
    
3. rpc_one_img_py  
    In this file, you need modify the test annotation file in following:
    
    with open('{your_test_annotation_file}', 'r') as f:  
    img ='{test_image_file}'  
    out_file='{result_out_file}'  
    
--------------------------
## Train

We trian our model in one 2080Ti card, and the command is:  

    python tools/train.py {config file}  
and an example is:  

    python tools/train.py configs/rpc/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll.py  
  
--------------------------
## Test

Our test command is:  

     python tools/test.py {config file} {checkpoint_file} [--out {result_file}] [--eval bbox]  
and an example is:  

    python tools/test.py configs/rpc/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll.py \
           result/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll/latest.pth --out \
           result/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll/result.pkl --eval bbox  
  
--------------------------
## Evaluate

Since 4 metrics other than mAP50 and mmAP are used, we perform the computation of these 4 metrics separately for the result file, i.e. rpc_eval.py.  
This evaluation file can print 4 metric values for all clutter patterns. And the command is:  

     python rpc_eval.py [--root {result_root_path}]  
and an example is:  

    python rpc_eval.py --root ./result/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll/  
  
--------------------------
## Note

If you want to train, test and evaluate the model sequentially at once, you can run the "rpc_run.py" file directly. 

    python rpc_run.py
If you want to use other configs, you only need modify the corresponding file names or paths in the "rpc_run.py" file.  

