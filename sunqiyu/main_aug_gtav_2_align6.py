from grpc import alts_channel_credentials
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, SYNTHIA, GTAV
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

import cv2
from torchvision.transforms.functional import normalize
import torchvision.transforms.functional as Fun
import copy

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'SYNTHIA'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--phase", type=str, default='support',
                        choices=available_models)

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    train_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtRandomScale(scale_range=[0.6, 2.1]),
        et.ExtRandomRotation(degrees=[0,30]),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_query_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtRandomScale(scale_range=[0.8, 2]),
        et.ExtRandomRotation(degrees=[0,30]),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_cs_dst = Cityscapes(root=opts.data_root,
                            split='train', transform=train_transform)
    val_cs_dst = Cityscapes(root=opts.data_root,
                            split='val', transform=val_transform)

    train_syn_dst = SYNTHIA(list_path='/root/Desktop/DeepLabV3Plus-Pytorch-master/datasets/data/SYNTHIA/train.txt',transform=train_transform)
    val_syn_dst = SYNTHIA(list_path='/root/Desktop/DeepLabV3Plus-Pytorch-master/datasets/data/SYNTHIA/train.txt',transform=train_query_transform)

    train_gtav_dst = GTAV(list_path='/root/Desktop/DeepLabV3Plus-Pytorch-master/datasets/data/GTAV/train.txt',transform=train_transform)

    return train_cs_dst, val_cs_dst, train_gtav_dst, val_syn_dst
    # return train_dst, val_dst



def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # outputs = model(images)
            outputs,_,_  = model(images)
            outputs = F.interpolate(outputs, size=images.shape[-2:], mode='bilinear', align_corners=False)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'synthia':
        opts.num_classes = 19
        
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_cs_dst, val_cs_dst, train_syn_dst, val_syn_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_syn_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    train_query_loader = data.DataLoader(
        val_syn_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_cs_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_syn_dst), len(val_cs_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    interval_cons_loss = 0
    interval_false_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        # for (support, query) in zip(train_loader,train_query_loader):
        for (images, labels) in train_loader:
            cur_itrs += 1
            
            trans_images = copy.deepcopy(images)
                        
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            
            
            optimizer.zero_grad()
            trans_images = transfer_lab(images).to(device, dtype=torch.float32)
            
            outputs_raw, f_out, f_low= model(images)
            outputs = F.interpolate(outputs_raw, size=images.shape[-2:], mode='bilinear', align_corners=False)
            
            trans_outputs_raw, t_f_out, t_f_low = model(trans_images)
            trans_outputs = F.interpolate(trans_outputs_raw, size=images.shape[-2:], mode='bilinear', align_corners=False)
            
            outputs_max, id = torch.max(torch.cat((outputs.unsqueeze(0),trans_outputs.unsqueeze(0)),0),0) 
            loss_seg_out = criterion(outputs, labels)
            loss_seg_trans = criterion(trans_outputs, labels)
            loss_seg = criterion(outputs_max, labels)
            
            #feature_align
            vectors, ids, false_vectors, false_ids = calculate_mean_vector(f_out.detach(), outputs_raw,  labels=labels)
            # class_vectors = update_objective_SingleVector(ids, vectors).to(device) 
            # class_vectors = nn.functional.normalize(class_vectors, p=2, dim=1)
            # false_class_vectors = update_objective_SingleVector(false_ids, false_vectors).to(device) 
            # false_class_vectors = nn.functional.normalize(false_class_vectors, p=2, dim=1)
            
            t_vectors, t_ids, false_t_vectors, false_t_ids = calculate_mean_vector(t_f_out.detach(), trans_outputs_raw, labels=labels)
            # t_class_vectors = update_objective_SingleVector(t_ids, t_vectors).to(device) #[19,256]
            # t_class_vectors = nn.functional.normalize(t_class_vectors, p=2, dim=1)
            # false_t_class_vectors = update_objective_SingleVector(false_t_ids, false_t_vectors).to(device) #[19,256]
            # false_t_class_vectors = nn.functional.normalize(false_t_class_vectors, p=2, dim=1)
            all_vectors = vectors + t_vectors
            all_ids = ids + t_ids
            class_features = np.array(list(constractive(all_ids, all_vectors).values()))
            
            feature_loss = 0
            feature_mean = []
            for i in range(len(class_features)):
                if len(class_features[i])<=1:
                    feature_mean.append(0)
                    continue
                else:
                    class_feature = torch.tensor(np.array(class_features[i]))
                    class_feature = nn.functional.normalize(class_feature, p=2, dim=1)
                    # for i in range(len(class_feature)):
                    #     a = class_feature[i]
                    #     for j in range(len(class_feature)-i-1):
                    #         b = class_feature[i+j+1]
                    #         feature_loss += 0.001 * torch.sum((torch.from_numpy(a)-torch.from_numpy(b))**2).to(device) 
                    mean = torch.mean(class_feature, dim=0)
                    feature_mean.append(mean)
                    for feature in class_feature:
                        feature_loss += torch.sum((feature-mean)**2)
                                    
            loss = 0.5 * (loss_seg_out + loss_seg_trans + loss_seg) + feature_loss
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            np_cons_loss = feature_loss.detach().cpu().numpy()
            # np_false_loss = (feature_false_loss + low_feature_false_loss).detach().cpu().numpy()
            interval_loss += np_loss
            interval_cons_loss += np_cons_loss
            # interval_false_loss += np_false_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                interval_cons_loss = interval_cons_loss / 10
                interval_false_loss = interval_false_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f, Feature_loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss, interval_cons_loss))
                interval_loss = 0.0
                interval_cons_loss = 0.0
                # interval_false_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = val_cs_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = val_cs_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return

def transfer_lab(src):
    src = src.detach().cpu().numpy()
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    
    # img = res[0]
    # img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
               
    trans_img = []
    
    for j in range(len(src)):
        img = src[j]
        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # style_index = random.randint(0, len(src)-1)
        # img_style = src[style_index]
        # img_style = (denorm(img_style) * 255).transpose(1, 2, 0).astype(np.uint8)
        # img_style = cv2.cvtColor(img_style, cv2.COLOR_RGB2LAB)
        t_mean = [random.uniform(0, 255), random.uniform(10, 180), random.uniform(10, 180)] 
        t_std = [random.uniform(0, 150), random.uniform(0, 30), random.uniform(0, 30)]
        # t_mean = [img_style[:,:,0].mean(), img_style[:,:,1].mean(), img_style[:,:,2].mean()] 
        # t_std = [img_style[:,:,0].std(), img_style[:,:,1].std(), img_style[:,:,2].std()]
        for i in range(3):
            mean, std = img[:,:,i].mean(),img[:,:,i].std()
            img[:,:,i] = ((img[:,:,i]-mean)/std) * t_std[i] + t_mean[i]
            # np.clip(img, 0, 255)
        # img_bgr = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        # raw_img = (denorm(res[j]) * 255).transpose(1, 2, 0).astype(np.uint8)
        # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        # plot_image = np.hstack((img_bgr, raw_img))
        # cv2.imwrite('plot/'+str(int(cur_itrs))+'_'+str(j)+'.png', plot_image)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        img = Fun.to_tensor(np.array(img))
        img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans_img.append(img.unsqueeze(0))
                
    # trans_img=np.concatenate(np.expand_dims(trans_img, axis = 0), 0)  

    trans_img = torch.cat(trans_img, 0)  
 
    
        # img_bgr = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        # raw_img = (denorm(res[j]) * 255).transpose(1, 2, 0).astype(np.uint8)
        # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        # plot_image = np.hstack((img_bgr, raw_img))
        # cv2.imwrite('plot/'+str(int(cur_itrs))+'_'+str(j)+'.png', plot_image)
        # cv2.imshow('1', plot_image)
        # cv2.waitKey(0)
    return trans_img



def calculate_mean_vector(feat_cls, outputs, labels=None):
    labels = Fun.resize(labels, feat_cls.shape[-2:], Image.NEAREST)
    outputs_softmax = F.softmax(outputs, dim=1)
    outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
    outputs_argmax = process_label(outputs_argmax)
    # if labels is None:
    #     outputs_pred = outputs_argmax
    # else:
    labels = labels.unsqueeze(1)
    labels_expanded = process_label(labels)
    # outputs_pred = labels_expanded * outputs_argmax
    true_id = torch.where(outputs_argmax==labels_expanded, 1, 0)
    outputs_pred = labels_expanded * true_id
    scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
    
    false_id = torch.where(outputs_argmax==labels_expanded, 0, 1)
    false_pred = labels_expanded * false_id
    false_scale_factor = F.adaptive_avg_pool2d(false_pred, 1)
    vectors = []
    ids = []
    false_vectors = []
    false_ids = []
    for n in range(feat_cls.size()[0]):
        for t in range(19):
            if scale_factor[n][t].item()==0:
                continue
            if (outputs_pred[n][t] > 0).sum() < 10:
                continue
            if false_scale_factor[n][t].item()==0:
                continue
            if (false_pred[n][t] > 0).sum() < 10:
                continue
            s = feat_cls[n] * outputs_pred[n][t]            
            s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
            vectors.append(s)
            ids.append(t)
            
            false_s = feat_cls[n] * false_pred[n][t]
            false_s = F.adaptive_avg_pool2d(false_s, 1) / false_scale_factor[n][t]
            false_vectors.append(false_s)
            false_ids.append(t)
    return vectors, ids, false_vectors, false_ids

def process_label(label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch, channel, w, h = label.size()
    pred1 = torch.zeros(batch, 19 + 1, w, h).to(device)
    a = torch.LongTensor([19]).to(device)
    id = torch.where(label < 19, label, a)
    pred1 = pred1.scatter_(1, id.long(), 1)
    return pred1

def update_objective_SingleVector(ids, vectors):

    objective_vectors = torch.zeros([19, 256])
    objective_vectors_num = torch.zeros([19])
    
    for t in range(len(ids)):
        id = ids[t]
        vector = vectors[t].detach().cpu().numpy()

        objective_vectors[id] = objective_vectors[id] * objective_vectors_num[id] + vector.squeeze()
        objective_vectors_num[id] += 1
        objective_vectors[id] = objective_vectors[id] / objective_vectors_num[id]
        objective_vectors_num[id] = min(objective_vectors_num[id], 3000)
    
    return objective_vectors

def constractive(ids, vectors):
    
    # objective_vectors = torch.zeros([19, 256])
    # objective_vectors_num = torch.zeros([19])
    
    class_feature = {i:[] for i in range(19)}
    
    for t in range(len(ids)):
        id = ids[t]
        vector = vectors[t].detach().cpu().numpy()
        class_feature[id].append(vector)
    
    return class_feature
 
def calculate_fuse_feature(query_outputs, class_vectors, query_feat): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              
    query_outputs_trans = query_outputs.permute(0,2,3,1) #[4,600,600,19]
    c = torch.matmul(query_outputs_trans, class_vectors) #[4,600,600,256]
    c = c.permute(0,3,1,2)             #[4,256,600,600]
    fuse_feat = torch.cat( [ c, query_feat ], dim=1 )
    conv = nn.Conv2d(512, 19, 1).to(device)
    query_outputs = conv(fuse_feat)  
    return query_outputs        




if __name__ == '__main__':
    main()
