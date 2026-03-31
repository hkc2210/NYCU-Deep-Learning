Project structure follows the Lab 2 guideline.

Main files
- src/models/unet.py: UNet model
- src/models/resnet34_unet.py: ResNet34 + UNet model
- src/oxford_pet.py: dataset loading and preprocessing
- src/utils.py: metrics, TTA, RLE, and helper functions
- src/train.py: training code
- src/evaluate.py: validation and threshold search
- src/inference.py: test-time inference and submission generation

Dataset
- dataset/oxford-iiit-pet/ stores the Oxford-IIIT Pet Dataset
- data_list_unet/ stores train.txt, val.txt, and test_unet.txt for the UNet competition
- data_list_resnet/ stores train.txt, val.txt, and test_res_unet.txt for the ResNet34_UNet competition

How to run inference
1. Make sure the required checkpoint file exists in saved_models/.
2. Run inference from the project root.

Example: ResNet34_UNet
python src/inference.py --model resnet34_unet --checkpoint saved_models/resnet34_unet_aspp_ds_img320_lr0p0001_wd0p0001.pth --data-root dataset/oxford-iiit-pet --split-root data_list_resnet --test-file test_res_unet.txt --img-size 320 --batch-size 16 --submission-path saved_models/resnet34_submission.csv

Example: UNet
python src/inference.py --model unet --checkpoint saved_models/unet_bc64_img320_lr0p0001_wd0p0001.pth --data-root dataset/oxford-iiit-pet --split-root data_list_unet --test-file test_unet.txt --img-size 320 --batch-size 8 --base-channels 64 --submission-path saved_models/unet_submission.csv

Notes
- All test-time inference logic is placed in src/inference.py.
- The test split is different for UNet and ResNet34_UNet, so the correct split directory and test file must be used.
- Predicted masks are resized back to the original image size before RLE encoding.
