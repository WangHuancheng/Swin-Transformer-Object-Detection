
# The new config inherits a base config to highlight the necessary modification
_base_ = '../swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('neuron',)
data = dict(
    train=dict(
        img_prefix='train/',
        classes=classes,
        ann_file='train/train.json'),
    val=dict(
        img_prefix='val/',
        classes=classes,
        ann_file='val/val.json'),
    test=dict(
         img_prefix='val/',
         classes=classes,
         ann_file='val/val.json'))
optimizer_config = dict(grad_clip=None)
