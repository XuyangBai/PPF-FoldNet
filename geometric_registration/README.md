Dictionary Structure
- gt_result: save the ground truth (gt.log and gt.info) provided by 3dmatch dataset. each scene will have a dictionary.
  - sun3d-hotel_umd-maryland_hotel3-evaluation
    - gt.log
    - gt.info
- pred_result: save the predicted result. each scene will have a dictionary.
 - sun3d-hotel_umd-maryland_hotel3
   - 3dmatch_result: save the per pair registration result using 3dmatch descriptor..
   - ppf_result: save the per pair registration results using ppf descriptor.
