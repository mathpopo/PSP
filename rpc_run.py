import os

if __name__ == '__main__':
    cmd = 'python tools/train.py configs/rpc/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll.py'
    os.system(cmd)
    cmd = 'python tools/test.py configs/rpc/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll.py ' \
          'result/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll/latest.pth --out ' \
          'result/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll/result.pkl --eval bbox '
    os.system(cmd)
    cmd = 'python rpc_eval.py --root ./result/faster_rcnn_r50_fpn_3x_rpc_protoS_srr_mll/'
    os.system(cmd)
