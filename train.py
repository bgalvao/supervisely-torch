# DONE: augmentations
# TODO: checkpoints
# TODO: logging; tensorboard; 
# TODO: hyperparams D:

# for loading your custom dataset
from supervisely2pytorch import SuperviselyDataset, get_num_classes

# for spawning the model and more
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as pt_transforms
from torch.utils.tensorboard import SummaryWriter

# training setup utils
from references.engine import train_one_epoch, evaluate
from references import transforms as T
from references.transforms import TransformWrapper as Wrapper
from references import utils

# texting logs
from notifications.telegram import send_message as telegram
import os
import datetime


# deduced params
_NUM_CLASSES = get_num_classes()  # reads from dataset and sets this constant

# params
NUM_EPOCHS = 10
BATCH_SIZE = 12  # as great as your GPU can handle
ALT_BACKBONE = None
TRAIN_SPLIT = .95  # make sure it returns an uint index

LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

LOG = True


def exit():
    from sys import exit
    exit(-1)

if LOG:
    # then start tensorboard summary writer
    # log_dir = './runs/logs'
    writer = SummaryWriter()
    writer.add_scalar('ass/hole', 11)
    # get the run name
    run_name = writer.log_dir.split('/')[-1]

    # model_save_path = writer.log_dir.split()
    checkpoints_dir = './runs/models'
    model_save_dir = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, 'model.pt')



def get_transforms(train=False):
    
    transforms = []
    
    # transforms on PIL
    if train:
        transforms.append(
            Wrapper(pt_transforms.ColorJitter(
                    brightness=.3, contrast=.5, saturation=.4, hue=.4
        )))
    
    transforms.append(T.ToTensor())
    # transforms on Tensor!!
    if train:
        transforms.append(T.RandomHorizontalFlip(prob=.5))
        # transforms.append(Wrapper(pt_transforms.RandomErasing(.3)))
    
    transforms.append(Wrapper(pt_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )))
    
    return T.Compose(transforms)


def spawn_model(alt_backbone=ALT_BACKBONE):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True
    )
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, _NUM_CLASSES)
    return model



if __name__ == "__main__":

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() \
             else torch.device('cpu')

    # use our dataset and defined transformations
    # you must at least pass a toTensor() transform!
    dataset = SuperviselyDataset(transforms=get_transforms(train=True))
    dataset_test = SuperviselyDataset(transforms=get_transforms())

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    TRAIN_SPLIT = int(TRAIN_SPLIT * (len(dataset) - 1))
    dataset = torch.utils.data.Subset(dataset, indices[:TRAIN_SPLIT])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[TRAIN_SPLIT:])

    # define training and validation data loaders
    num_workrs = 1
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=num_workrs,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=num_workrs,
        collate_fn=utils.collate_fn
    )

    # move model to the right device
    model = spawn_model()
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )


    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    start = datetime.datetime.now()

    for epoch in range(NUM_EPOCHS):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10
        )
        
        # update the learning rate
        lr_scheduler.step()
        
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        # uncomment if block if you want to save checkpoints
        # if (epoch+1 % 2) == 0:
        #     torch.save(
        #         {
        #             'epoch': epoch+1,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             # 'loss': loss,
        #             # 'val_loss': val_loss
        #         },
        #         os.path.join(model_save_dir, 'model_{}.pt'.format(epoch+1))
        #     )


    end = datetime.datetime.now()
    duration = end-start

    # telegram("\n".join([
    #     "- training job that started at {}".format(
    #         str(start).split('.')[0].replace(' ', '@')
    #     ),
    #     "- is now finished at {}".format(formatted_end_dt),
    #     "- took {} long".format(str(end - start).split('.')[0])
    # ]))
  
    torch.save(model.state_dict(), model_save_path)
    print('swweet')

    if LOG:
        writer.add_hparams(
            hparam_dict={
                'total_epochs':NUM_EPOCHS,
                'batch_size':BATCH_SIZE,
                'train_split':TRAIN_SPLIT,
                'duration':duration,

                'learning_rate':LEARNING_RATE,
                'momentum':MOMENTUM,
                'weight_decay':WEIGHT_DECAY,

                'num_classes':_NUM_CLASSES
            },
            metric_dict=None
        )