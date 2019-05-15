## Dictionary Structure
- `gt_result/`: save the ground truth (gt.log and gt.info) provided by 3dmatch dataset. each scene will have a dictionary.
  - `sun3d-hotel_umd-maryland_hotel3-evaluation/`
    - gt.log
    - gt.info
- `pred_result/`: save the predicted result. each scene will have a dictionary.
  - `sun3d-hotel_umd-maryland_hotel3/`
    - `3dmatch_result/`: save the per pair registration result using 3dmatch descriptor.
    - `ppf_result/`: save the per pair registration results using ppf descriptor.

## How to run the code

### Prepare the keypoint feature descriptors

```bash
python preparation.py 05121612 
```

The parameter is the experiment id of the model you want to load. The result will be saved in the folder `ppf_desc_05121612/`

### Evaluate the feature descriptors on 3DMatch benchmark

```bash
python evaluate_ppfnet.py 05121612
```

Still the parameter is the experiment id. The result will be saved in `pred_result/` folder