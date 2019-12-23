# for loading your custom dataset
from supervisely2pytorch import SuperviselyDataset, get_num_classes

# for spawning the model and more
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# training setup utils
from references.engine import train_one_epoch, evaluate
from references import transforms as T
from references import utils

# deduced params
_NUM_CLASSES = get_num_classes()  # reads from dataset and sets this constant

# params
NUM_EPOCHS = 10
BATCH_SIZE = 12
ALT_BACKBONE = None
TRAIN_SPLIT = int(20)  # make sure it returns an uint index


def get_transform(hflip=False):
    transforms = []
    # transforms.append(T.ToTensor())
    if hflip:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
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

    # t = get_transform(True)

    # dataset = SuperviselyDataset()
    # sample = dataset[1]
    # print(t)
    # print(sample)
    # print(t(sample[0]))

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() \
             else torch.device('cpu')

    # use our dataset and defined transformations
    # you must at least pass a toTensor() transform!
    # TODO: normalize?
    t = get_transform()
    print(hex(id(t)))
    dataset = SuperviselyDataset(transforms=t)
    dataset_test = SuperviselyDataset(transforms=t)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:TRAIN_SPLIT])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[TRAIN_SPLIT:])


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)


    # move model to the right device
    model = spawn_model()
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    for epoch in range(NUM_EPOCHS):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10
        )
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
