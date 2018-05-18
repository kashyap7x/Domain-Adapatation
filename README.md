# Domain Adaptation

- ResNet-50 baselines for [Office31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
- Synthetic gradient module training added
- Filtering for synthetic gradients by voting added
- Adversarial filtering added
- PSPNet baseline for CityScapes added
- Pretrined segmentation model loading done (place in `segmentation/pretrained`)
- Full-scale testing for segmentation (different batch size)
- Added code for adaptation of segmentation (filtered gradients) 
- Eval code for mIoU, class IoUs

## TODO

- Logging and plots for Office31
- New metrics (% filtered gradients, accuracy of filtered gradients)
- Hard mining and weighted loss