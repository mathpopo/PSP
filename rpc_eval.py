import pickle
import json
import numpy as np
from argparse import ArgumentParser
import torch
import os

if __name__ == '__main__':
    parser = ArgumentParser(description="RPC eval")
    parser.add_argument('--ann_file', type=str,
                        default='/media/xxx/Data/RPC/instances_test2019.json')
    parser.add_argument('--root', type=str, default='./result/faster_rcnn_r50_fpn_3x_rpc_protoS_ranking_mll/')
    parser.add_argument('--result_name', type=str, default='result')
    args = parser.parse_args()
    file_dir = args.root + args.result_name + '.pkl'
    file = open(file_dir, 'rb')
    result = pickle.load(file)
    file.close()
    with open(args.ann_file, 'r') as f:
        data = json.load(f)
        f.close()

    num_classes = 200

    GT = np.zeros([len(result), num_classes], dtype=int)
    P = np.zeros([len(result), num_classes], dtype=int)
    img_idx = np.zeros([len(result)], dtype=int)
    img_idx_easy = []
    img_idx_mid = []
    img_idx_hard = []

    for i in range(len(result)):
        img_idx[i] = data['images'][i]['id']
        if data['images'][i]['level'] == 'easy':
            img_idx_easy.append(i)
        elif data['images'][i]['level'] == 'medium':
            img_idx_mid.append(i)
        elif data['images'][i]['level'] == 'hard':
            img_idx_hard.append(i)
        for j in range(num_classes):
            P[i][j] = result[i][j].shape[0]

    for i in range(len(data['annotations'])):
        if num_classes == 200:
            ann_id = data['annotations'][i]['category_id'] - 1
        else:
            ann_id = cls.index(data['annotations'][i]['category_id'])
        img_id = data['annotations'][i]['image_id']
        x = np.where(img_idx == img_id)
        GT[x[0][0]][ann_id] += 1
    CD = P - GT
    CD = np.abs(CD)

    idx = np.nonzero(np.sum(CD, axis=1))
    np.save(args.root+'fp_idx.npy', idx[0])

    # cAcc
    cAcc = 1 - (np.count_nonzero(np.sum(CD, axis=1)) / len(result))
    cAcc_easy = 1 - (np.count_nonzero(np.sum(CD[img_idx_easy], axis=1)) / len(img_idx_easy))
    cAcc_mid = 1 - (np.count_nonzero(np.sum(CD[img_idx_mid], axis=1)) / len(img_idx_mid))
    cAcc_hard = 1 - (np.count_nonzero(np.sum(CD[img_idx_hard], axis=1)) / len(img_idx_hard))

    # ACD
    ACD = np.sum(CD) / len(result)
    ACD_easy = np.sum(CD[img_idx_easy]) / len(img_idx_easy)
    ACD_mid = np.sum(CD[img_idx_mid]) / len(img_idx_mid)
    ACD_hard = np.sum(CD[img_idx_hard]) / len(img_idx_hard)

    # mCCD
    mCCD = np.sum(np.sum(CD, axis=0) / np.sum(GT, axis=0)) / num_classes
    mCCD_easy = np.sum(np.sum(CD[img_idx_easy], axis=0) / np.sum(GT[img_idx_easy], axis=0)) / num_classes
    mCCD_mid = np.sum(np.sum(CD[img_idx_mid], axis=0) / np.sum(GT[img_idx_mid], axis=0)) / num_classes
    mCCD_hard = np.sum(np.sum(CD[img_idx_hard], axis=0) / np.sum(GT[img_idx_hard], axis=0)) / num_classes

    # mCIoU
    mCIoU = 0
    mCIoU_easy = 0
    mCIoU_mid = 0
    mCIoU_hard = 0
    for i in range(num_classes):
        mCIoU += np.minimum(P[:, i], GT[:, i]).sum() / np.maximum(P[:, i], GT[:, i]).sum()
        mCIoU_easy += np.minimum(P[img_idx_easy, i], GT[img_idx_easy, i]).sum() / np.maximum(P[img_idx_easy, i],
                                                                                             GT[img_idx_easy, i]).sum()
        mCIoU_mid += np.minimum(P[img_idx_mid, i], GT[img_idx_mid, i]).sum() / np.maximum(P[img_idx_mid, i],
                                                                                          GT[img_idx_mid, i]).sum()
        mCIoU_hard += np.minimum(P[img_idx_hard, i], GT[img_idx_hard, i]).sum() / np.maximum(P[img_idx_hard, i],
                                                                                             GT[img_idx_hard, i]).sum()
    mCIoU /= num_classes
    mCIoU_easy /= num_classes
    mCIoU_mid /= num_classes
    mCIoU_hard /= num_classes

    print('-----| Level  | cAcc   |  ACD  |  mCCD | mCIoU  |-----')
    print('-----| Easy   | {:>5.2f}% | {:>5.2f} | {:>5.2f} | {:>5.2f}% |-----'.format(cAcc_easy * 100, ACD_easy, mCCD_easy, mCIoU_easy * 100))
    print('-----| Medium | {:>5.2f}% | {:>5.2f} | {:>5.2f} | {:>5.2f}% |-----'.format(cAcc_mid * 100, ACD_mid, mCCD_mid, mCIoU_mid * 100))
    print('-----| Hard   | {:>5.2f}% | {:>5.2f} | {:>5.2f} | {:>5.2f}% |-----'.format(cAcc_hard * 100, ACD_hard, mCCD_hard, mCIoU_hard * 100))
    print('-----| Avg    | {:>5.2f}% | {:>5.2f} | {:>5.2f} | {:>5.2f}% |-----'.format(cAcc * 100, ACD, mCCD, mCIoU * 100))

    # f = open(args.root+'result_{:.4f}'.format(cAcc)+'.pkl', 'wb')
    # pickle.dump(result, f)
    # f.close()
