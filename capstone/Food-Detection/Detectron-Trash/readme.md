# Food Detection 🥪

Detectron2-based model trained on the [2021 AICROWD Dataset](https://www.aicrowd.com/challenges/food-recognition-challenge/dataset_files)

The model was trained on 1 Tesla GPU (AWS.G4DN.XLarge) for roughly 10 hours

## 🔨 Data Processing
Step zero: Download the training/testing data from the official website

Discover: some images annotations have the wrong dimensions

Step one: Run dataproc.ipynb to fix the annotations in train, val respectively

Step two: You should end up with train.json, test.json. Move those to the right folders


## 📋 Steps to train your OWN model
**Step zero: An NVIDIA GPU with CUDA installed.** Below is some helpful code to find your CUDA version. 

```
import torch

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
```

Step one: Run the first few cells of model.ipynb to be acquainted with detectron/make sure the notebook works

Step two: Make sure you **did not** download **model_final.pth**, and set `trainer.resume_or_load(resume = False)`

Step three: Run the big chunk of text that has a config (the one with `trainer.train()`)

Step four: Run the validation tests after training has completed

## 👑 Steps to just run a model
Step one: Go to the outputs folder and follow steps to download **model_final.pth**

Step two: Make sure to move **model_final.pth** to outputs folder

Step three: Set up your config and predictor with
```
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train11",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER =  150000 # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 273  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
```
Step four: Predict on the test examples with
```
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("val11", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "val11")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
```
Step five: Enjoy your model!

## 🤔 QNA
Q: Why is the config the way it is?

A: Batch-Size = 1 in mask-rcnn architecture. I slowly increased the number of iterations as I saw the loss reducing

Q: What is an MASK-RCNN?

A: Idk

Q: Why is "cls_acc" so high but the model isn't performing with 90% accuracy on actual predictions?

A: Class Accuracy measures "accuracy of class prediction given that an object is present in the box" (dasturge)

Q: Where do you think the model "gets it wrong"?

A: The image classification and bounding box sections. Some images have many objects to detect and the model is unable to make solid predictions in those cases. Moreover, many images have ambigiuous labels (bread, bread-5-grain, brain-grain), and some bounding boxes are not in the right positions. Finally, some images are just confusing.



