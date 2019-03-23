Tommy:
-  [x] Produce a tabular dataset of precip / temp / vegetation health indices
- [x] upload to github (if < 50mb) (SENT BY WE TRANSFER)
- [ ] Mask out the sea values (not sure if already done - check!)
- [ ] Create a DataFrame CSVCleaner.create_anomaly() method to compute the
       Pixel-Year anomaly.
- [ ] NDVI anomaly won’t get returned by the readfile() method, because it’s not in VALUE_COLS, so we will need to add it there and to VEGETATION_COLS
- [ ] Add an argument for deciding which column is going to be the TARGET_COL

Gabriel:
- [x] produce project skeleton
- [x] Mask out the LST values == 200
