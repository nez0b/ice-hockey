from .models import Detector, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    print('device: ', device)

    model = model.to(device)
    if args.continue_training: 
        print("continue training")
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    #loss = torch.nn.L1Loss()
    #loss = torch.nn.MSELoss()
    det_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    size_loss = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    import inspect
    #transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    #train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    transform = dense_transforms.Compose([dense_transforms.ColorJitter(0.2, 0.5, 0.5, 0.2), 
                                          dense_transforms.RandomHorizontalFlip(), 
                                          dense_transforms.ToTensor(),
                                          dense_transforms.ToHeatmap()])
    train_data = load_data('test.pkl', num_workers=0, transform=transform, batch_size=10)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()

        for img, gt_det, gt_size in train_data:
            img, gt_det, gt_size = img.to(device), gt_det.to(device), gt_size.to(device)

            size_w, _ = gt_det.max(dim=1, keepdim=True)
            #img = torch.movedim(img, 3, 1)

            det, size = model(img)
            # Continuous version of focal loss
            #print('gt_det: ', gt_det.shape)
            #print('det: ', det.shape)
            p_det = torch.sigmoid(det * (1-2*gt_det))
            det_loss_val = (det_loss(det, gt_det)*p_det).mean() / p_det.mean()
            size_loss_val = (size_w * size_loss(size, gt_size)).mean() / size_w.mean()
            #loss_val = det_loss_val + size_loss_val * args.size_weight
            loss_val = det_loss_val + size_loss_val * 0.02

            #if train_logger is not None and global_step % 100 == 0:
            #    log(train_logger, img, gt_det, det, global_step)

            if train_logger is not None:
                train_logger.add_scalar('det_loss', det_loss_val, global_step)
                train_logger.add_scalar('size_loss', size_loss_val, global_step)
                train_logger.add_scalar('loss', loss_val, global_step)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if valid_logger is None or train_logger is None:
            print('epoch %-3d det loss %-3d  size loss %-3d' %
                  (epoch,det_loss_val,size_loss_val ) )
        save_model(model)

    
    save_model(model)

def log(logger, img, label, pred, global_step):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
