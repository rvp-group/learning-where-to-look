# Training

>[!IMPORTANT]
This README is not completed yet

## Prepare dataset

```bash
python3 generate_training_data.py --data_path <path-to-data> \
                                  --positive_percentage <percentage-of-positive-labels> \
                                  --rows <num-img-rows> \
                                  --cols <num-img-rows> \
                                  --output_data <path-to-output-data>```
```
>[!IMPORTANT]
The argument `positive_percentage` is suggested to `0.5` to balance **training data** and to `0` for **test data** (not altering/chopping dataset)

`<path-to-data>` requires multiple folders with the following structure

```
├── database.db --> database
├── sparse --> SfM model
├── images --> set of images
├── img_name_to_colmap_Tcw.txt
└── img_nm_to_colmap_cam.txt
<!-- TODO check sampling directions -->
```