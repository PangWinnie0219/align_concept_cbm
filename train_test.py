import os
import json
import argparse
import time
import pandas as pd
import numpy as np
import torch
from torch import nn
from datetime import datetime
import sklearn.metrics

from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from utils import utils
from utils import utils_torch
from utils.utils_torch import get_transforms
from dataset.att_label_dataset import AttLabelDataset
from dataset.unlabeled_image_dataset import UnlabeledImageDataset
from models.attribute_predictor import AttributePredictor
from models.linear_classifier import LinearClassifier
from models.non_linear_classifier import NonLinearClassifier, TwoLayerClassifier
from models.align_cbm import ConceptBottleneckModel
from loss_func.multi_task_loss import MultiTaskLoss

softmax = nn.Softmax(dim=1)


def setup_args():
    parser = argparse.ArgumentParser(description="WBC Aligning Concept Bottleneck Model Training")
    
    # datasets
    parser.add_argument(
        "--alpha_true",
        type=str,
        default='./dataset_txt/pbc_alpha_true_11.csv',
        help="file that define the concept importance from clinical knowledge.{2,1,0}, 2 represents the most important concept",
    )
    parser.add_argument(
        "--attribute",
        type=str,
        default="./dataset_txt/attribute.yml",
        help="attribute yaml file that defines the attribtue names and possible values",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="./dataset_txt/pbc_attr_v1_train.csv",
        help="training data",
    )
    parser.add_argument(
        "--image_dir",
        default="./data/PBC/",
        help="Root directory containing image files",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="./dataset_txt/pbc_attr_v1_val.csv",
        help="validation data",
    )
    parser.add_argument(
        "--image_dir_val",
        default=None,
        help="Root directory containing image files if different from train",
    )
    parser.add_argument(
        "--test",
        nargs="*",
        default=[
            "./dataset_txt/pbc_attr_v1_test.csv",
            "./dataset_txt/RaabinWBCTestA.csv",
            "./dataset_txt/scirep_test.csv",
        ],
        help="Paths to the test CSV files. Multiple paths possible.",
    )
    parser.add_argument(
        "--image_dir_test",
        nargs="*",
        default=["./data/PBC/", "./data/RaabinWBC/", "./data/scirep/"],
        help="Root directories for test images, corresponding to each test file path",
    )
    parser.add_argument(
        "--att_label_test",
        nargs="*",
        default=[True, True, False, False],
        help="Present of attribute annotations for test images, corresponding to each test file path",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="./dataset_txt/label.yml",
        help="yaml file of cell label dictionary",
    )
    parser.add_argument(
        "--attribute_binarized",
        type=str,
        default="./dataset_txt/attribute_binarized.yml",
        help="yaml file of  list of binarized attributes",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Column name for labels in the CSV file",
    )

    # sgd training
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64, 
        help="batch size"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.0001,  # 0.001
        help="initial learning rate"
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=-1,
        help="minimum learning rate to reach with cosine lr decay. -1 means no lr scheduler",
    )
    parser.add_argument(
        "--decay", 
        type=float, 
        default=0.01, 
        help="weight decay"
    )
    # args for lr scheduler
    parser.add_argument(
        '--scheduler', 
        type=str, 
        default='none', 
        choices=['cosine', 'step', "none"], 
        help='type of scheduler'
    )
    parser.add_argument(
        '--eta_min',
        type=float, 
        default=1e-6, 
        help='minimum learning rate for cosine annealing'
    )
    parser.add_argument(
        '--T_max', 
        type=int, 
        default=30, 
        help='maximum number of iterations for cosine annealing'
    )
    parser.add_argument(
        '--step_size', 
        type=int, 
        default=10, 
        help='period of learning rate decay for stepLR'
    )
    parser.add_argument(
        '--gamma', 
        type=float, 
        default=0.5, 
        help='multiplicative factor of learning rate decay'
    )
    # freeze epoch
    parser.add_argument(
        '--freeze_epoch', 
        type=int, 
        default=None, 
        help='epoch to freeze x->c networks to prevent overfitting (default: None)'
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs",
    )

    # loss
    parser.add_argument(
        "--lamda", 
        type=float, 
        default=1, 
        help="influential degree of alignment loss"
    )    
    parser.add_argument(
        "--phi", 
        type=float, 
        default=1, 
        help="influential degree of attribute prediction loss"
    )  
    parser.add_argument(
        "--dropout_rate", 
        type=float, 
        default=0, 
        help="influential degree of attribute prediction loss"
    )  

    # model
    parser.add_argument(
        "--backbone", 
        default="resnet50", 
        help="Choice of image encoder"
    )
    parser.add_argument(
        "--classifier", 
        default="linear", 
        help="Choice of classifer",
        choices=["linear", "nonlinear", "two_layer"],
    )
    parser.add_argument(
        "--num_hidden", 
        type=int, 
        default=128, 
        help="Number of hidden units"
    )
    parser.add_argument(
        "--layer_norm", 
        type=int, 
        default=0, 
        help="0 to disable layer_norm in classifier",
        choices=[0,1]
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="imagenet",
        help="pretrained weight of backbone",
    )

    # loss
    parser.add_argument(
        "--label_smoothing", 
        type=float, 
        default=0.05, 
        help="label smoothing"
    )

    # computation
    parser.add_argument(
        "--seed", 
        default=1, 
        type=int, 
        help="seed. -1 means random from time."
    )
    parser.add_argument(
        "--cudnn_deterministic", 
        type=int, 
        default=1, 
        help="cudnn_deterministic"
    )
    parser.add_argument(
        "--cudnn_benchmark", 
        type=int, 
        default=0, 
        help="cudnn_benchmark"
    )
    parser.add_argument(
        "--cudnn_enabled", 
        type=int, 
        default=1, 
        help="cudnn_enabled"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=8, 
        help="workers for torch.utils.data.DataLoader"
    )
    parser.add_argument(
        '--multi_gpu', 
        action='store_true', 
        help='Use multiple GPUs if available'
    )

    # logging and saving
    parser.add_argument(
        "--save_dir",
        default=None,
        required=False,
        help="Hard-coded directory to save the logs. Optional.  If not specified, --saveroot --saveprefix --saveargs will be used with datetime in the dir name.",
    )
    parser.add_argument(
        "--saveroot",
        default="./experiments/default",
        help="Root directory to make the output directory",
    )
    parser.add_argument(
        "--saveprefix",
        default="pbc",
        help="prefix to append to the name of log directory",
    )
    parser.add_argument(
        "--saveargs",
        default=["seed", "backbone","lamda"],
        nargs="+",
        help="args to append to the name of log directory",
    )
    parser.add_argument(
        "--save_last_model",
        action='store_true',
        help="save last model weights or not",
    )
    return parser.parse_args()


def get_test_dataloader(args, test, image_dir_test, attribute_encoders, label_encoder, att_label=True):
    df_test = pd.read_csv(test)
    if att_label:
        dataset_test = AttLabelDataset(df_test, image_dir=image_dir_test, 
                                    transform=get_transforms("test"), 
                                    attribute_encoders=attribute_encoders, 
                                    label_encoder=label_encoder['label'])
    else:
        dataset_test = UnlabeledImageDataset(df_test, image_dir=image_dir_test, 
                                transform=get_transforms("test"),  
                                label_encoder=label_encoder['label'])

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=(8 > 0),
        pin_memory=True,
        drop_last=False,
    )
    return dataloader_test


def get_test_results(dataset_name, dataloader_test, model, log, log_path, att_label=True, save_log=True):
    model.eval()
    y_pred_all = []
    y_gt_all = []
    c_pred_all = []
    c_gt_all = []
    with torch.no_grad():
        for data in dataloader_test:
            if att_label:
                images_val, c_gt, y_gt = data["image"].cuda(), data["attributes"].cuda(), data["y_label"].cuda()
                c_gt_all.append(c_gt)
            else:
                images_val, y_gt = data["image"].cuda(), data["y_label"].cuda()
            c_probs_list, y_probs = model(images_val, apply_softmax=True, get_delta_y=False)

            c_labels_list = [torch.argmax(c_probs, dim=1).cpu().numpy() for c_probs in c_probs_list]
            c_pred_all.append(c_labels_list)

            y_pred_all.append(torch.argmax(y_probs, dim=1).cpu().numpy())
            y_gt_all.append(y_gt)

        c_pred_all = [np.concatenate(x) for x in zip(*c_pred_all)]

        if len(c_gt_all) > 0:
            c_gt_all = torch.cat(c_gt_all, dim=0)
            c_f1_scores_list = [f1_score(true_labels.cpu().numpy(), predicted_labels, average='macro')
                                for true_labels, predicted_labels in zip(c_gt_all.t(), c_pred_all)]
            c_ave_f1_score = np.mean(c_f1_scores_list)
            c_accuracy_list = [accuracy_score(true_labels.cpu().numpy(), predicted_labels)
                               for true_labels, predicted_labels in zip(c_gt_all.t(), c_pred_all)]
            c_ave_accuracy = np.mean(c_accuracy_list)

        y_pred_all = np.concatenate(y_pred_all, axis=0)
        y_gt_all = torch.cat(y_gt_all, dim=0)  
        y_f1_score = f1_score(y_gt_all.cpu().numpy(), y_pred_all, average="macro")
        y_accuracy = accuracy_score(y_gt_all.cpu().numpy(), y_pred_all)

    print(f'Result for {dataset_name}:')
    if len(c_gt_all) > 0:
        print(f'Aver F1 score on att prediction: {c_ave_f1_score*100:.2f}%')
        print(f'Aver Accuracy on att prediction: {c_ave_accuracy*100:.2f}%')
    print(f'F1 score on class prediction: {y_f1_score*100:.2f}%')
    print(f'Accuracy on class prediction: {y_accuracy*100:.2f}%')
    print('========================================================')

    if save_log:
        if len(c_gt_all) > 0:
            log['test_' + dataset_name + '_' + 'c_f1_scores_list'] = c_f1_scores_list
            log['test_' + dataset_name + '_' + 'c_ave_f1_score'] = c_ave_f1_score
            log['test_' + dataset_name + '_' + 'c_ave_accuracy'] = c_ave_accuracy
        log['test_' + dataset_name + '_' + 'y_f1_score'] = y_f1_score
        log['test_' + dataset_name + '_' + 'y_accuracy'] = y_accuracy
        with open(log_path, 'w') as file:
            json.dump(log, file, indent=4)
    


def main(args):
    print(args)

    args.seed = utils.setup_seed(args.seed)
    utils_torch.make_deterministic(
            args.seed,
            cudnn_deterministic=args.cudnn_deterministic,
            cudnn_benchmark=args.cudnn_benchmark,
            cudnn_enabled=args.cudnn_enabled,
    )
    
    # logging
    if args.save_dir is None:
        log_dir = utils.setup_savedir(
            prefix=args.saveprefix,
            basedir=args.saveroot,
            args=args,
            append_args=args.saveargs,
        )
    else:
        log_dir = args.save_dir
        if os.path.exists(log_dir):
            print(
                f"Directory {log_dir} already exists. Please delete it or specify a different directory."
            )
            exit(1)
        os.makedirs(log_dir)
        print(f"Saving logs to {log_dir}")
    log = {}
    log["git"] = utils.check_gitstatus()
    log["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_metadata_path = os.path.join(log_dir, "log_metadata.json")
    utils.save_json(log, log_metadata_path)
    utils.save_args(log_dir, args)

    # Load alpha contribution matrix
    alpha_true = pd.read_csv(args.alpha_true, index_col=0)

    attribute_encoders = utils.load_yaml(args.attribute)
    attribute_decoders = {k: {v: k for k, v in v.items()} for k, v in attribute_encoders.items()}
    attribute_sizes = [len(attribute_encoders[key]) for key in attribute_encoders.keys()]
    att_bin_cols = utils.load_yaml(args.attribute_binarized)
    label_encoder = utils.load_yaml(args.label)
    labels = list(label_encoder[args.label_col].keys())
    attribute_names = list(attribute_encoders.keys())

    # dataset and dataloader
    attribute_dict = attribute_encoders

    if args.image_dir_val is None:
        args.image_dir_val = args.image_dir

    df_train = pd.read_csv(args.train)
    dataset_train = AttLabelDataset(
        df_train, 
        image_dir=args.image_dir, 
        transform=get_transforms("train"), 
        attribute_encoders=attribute_dict, 
        label_encoder=label_encoder[args.label_col]
    )

    df_val = pd.read_csv(args.val)
    dataset_val = AttLabelDataset(
        df_val, 
        image_dir=args.image_dir_val, 
        transform=get_transforms("test"), 
        attribute_encoders=attribute_dict, 
        label_encoder=label_encoder[args.label_col]
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
        pin_memory=True,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
        pin_memory=True,
        drop_last=False,
    )

    # setup image encoder
    image_encoder, image_encoder_output_dim = utils_torch.get_image_encoder(args.backbone, pretrained=args.pretrained)
    for param in image_encoder.parameters():
        param.requires_grad = True

    attribute_predictor = AttributePredictor(attribute_sizes, image_encoder_output_dim, image_encoder, dropout_rate=args.dropout_rate)
    
    # setup classifier
    layer_norm = False
    if args.layer_norm > 0:
        layer_norm=True
        print('Use Layer Norm for classifier')
    if args.classifier == 'linear':
        # classifier = NonNegativeSoftmaxRegression(in_features=len(att_bin_cols), num_classes=len(labels), labels=labels, attribute_sizes=attribute_sizes)
        classifier = LinearClassifier(in_features=len(att_bin_cols), num_classes=len(labels))
        print('Using linear classifier!')
    elif args.classifier == 'nonlinear':
        classifier = NonLinearClassifier(in_features=len(att_bin_cols), num_classes=len(labels), num_hidden=20)
        print('Using nonlinear classifier!')    # MLP(20)
    elif args.classifier == 'two_layer':
        classifier = TwoLayerClassifier(in_features=len(att_bin_cols), num_classes=len(labels), num_hidden=args.num_hidden, layer_norm=layer_norm)
        print('Using two_layer classifier!')    # MLP(128)
        
    model = ConceptBottleneckModel(attribute_predictor=attribute_predictor, classifier=classifier)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.cuda()


    # loss
    task_configs = []
    attribute_names = list(attribute_dict.keys())
    weight = 1.0 / len(attribute_names)
    for i in range(len(attribute_names)):
        task_configs.append(
            {
                "name": attribute_names[i],
                "loss_func": nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
                "weight": weight,
            }
        )
    criterion_c = MultiTaskLoss(task_configs)
    criterion_y = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_align = nn.L1Loss()

    # sgd training optimizer,  lr scheduler, etc
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.decay)

    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None


    history = {
        'epoch': [], 
        'training_loss_c': [], 
        'training_loss_y': [], 
        'training_loss_align': [], 
        'c_ave_f1_score': [],
        'y_f1_score':[]
    }

    print("Start training...")

    best_metric = 0.0
    best_epoch = 0
    for i in tqdm(range(args.epochs), dynamic_ncols=True):
        model.train()
        for data in dataloader_train:    
            images, c_gt, y_gt = data["image"].cuda(), data["attributes"].cuda(), data["y_label"].cuda()
            optimizer.zero_grad()

            c_logits, y_logits, delta_y = model(images)

            pbc_alpha_true_tensor = torch.tensor(alpha_true.values).cuda()          # [K, L]

            # Expand y_gt to match the shape of delta_y for indexing
            y_gt_expanded = y_gt.unsqueeze(1).unsqueeze(2).expand(delta_y.size())   # [B, K, L]

            # Use gathered_pbc_alpha_true_tensor to index pbc_alpha_true_tensor based on y_gt
            gathered_pbc_alpha_true_tensor = torch.gather(pbc_alpha_true_tensor.unsqueeze(0).expand(delta_y.size(0), -1, -1), 1, y_gt_expanded)     # [B, K, L]

            # Mask delta_y based on pbc_alpha_true_tensor
            mask_low = (gathered_pbc_alpha_true_tensor == 0)    # [B, K, L]
            low_delta_y = delta_y[mask_low]
            mask_high = (gathered_pbc_alpha_true_tensor == 2)
            high_delta_y = delta_y[mask_high]

            loss_align_low = criterion_align(low_delta_y, torch.zeros_like(low_delta_y))
            loss_align_high = criterion_align(high_delta_y, torch.ones_like(high_delta_y))

            # do this for empty loss_align
            if torch.isnan(loss_align_low):
                loss_align_low = torch.tensor(0., device=loss_align_low.device)
            if torch.isnan(loss_align_high):
                loss_align_high = torch.tensor(0., device=loss_align_high.device)

            loss_align = loss_align_low + loss_align_high

            # cbm loss
            loss_c = criterion_c(c_logits, c_gt.t())
            loss_y = criterion_y(y_logits, y_gt)
            
            loss = (args.phi*loss_c) + (loss_y) + (args.lamda*loss_align)
    

            history['training_loss_c'].append((args.phi*loss_c).item())
            history['training_loss_y'].append(loss_y.item() if loss_y is not None else None)
            history['training_loss_align'].append((args.lamda*loss_align).item() if loss_align is not None else None)
            loss.backward()
            optimizer.step()
            
            # epoch freeze
            if args.freeze_epoch is not None and i == args.freeze_epoch:
                # freeze x->c networks 
                if hasattr(model, 'module'):
                    # Using nn.DataParallel
                    for param in model.module.attribute_predictor.parameters():
                        param.requires_grad = False
                else:
                    for param in model.attribute_predictor.parameters():
                        param.requires_grad = False

            # learning rate decay
            if scheduler is not None:
                scheduler.step()

        model.eval()
        y_pred_all = []
        y_gt_all = []
        c_pred_all = []
        c_gt_all = []
        with torch.no_grad():
            for data in dataloader_val:    
                images_val, c_gt, y_gt = data["image"].cuda(), data["attributes"].cuda(), data["y_label"].cuda()
                c_probs_list, y_probs = model(images_val, apply_softmax=True, get_delta_y=False)

                c_labels_list = [torch.argmax(c_probs, dim=1).cpu().numpy() for c_probs in c_probs_list]
                c_pred_all.append(c_labels_list)
                c_gt_all.append(c_gt)

                y_pred_all.append(torch.argmax(y_probs, dim=1).cpu().numpy() )
                y_gt_all.append(y_gt)
                
            c_pred_all = [np.concatenate(x) for x in zip(*c_pred_all)]
            c_gt_all = torch.cat(c_gt_all, dim=0) 
            c_f1_scores_list = [sklearn.metrics.f1_score(true_labels.cpu().numpy(), predicted_labels, average='macro') 
                    for true_labels, predicted_labels in zip(c_gt_all.t(), c_pred_all)]
            c_ave_f1_score = np.mean(c_f1_scores_list)
            
            y_pred_all = np.concatenate(y_pred_all, axis=0)
            y_gt_all = torch.cat(y_gt_all, dim=0)       
            y_f1_score = sklearn.metrics.f1_score(y_gt_all.cpu().numpy(), y_pred_all, average="macro") 

            history['epoch'].append(i)
            history["c_ave_f1_score"].append(c_ave_f1_score)
            history['y_f1_score'].append(y_f1_score)  
        
        if y_f1_score > best_metric:
            best_metric = y_f1_score
            best_epoch = i
            best_model_path = os.path.join(log_dir, "model_bestval.pth")
            other_info = {
                "backbone": args.backbone,
                'pretrained': args.pretrained,
                'classifier': args.classifier,
                "best_epoch": best_epoch,
                "best_metric": y_f1_score,
            }
            utils_torch.save_checkpoint(
                best_model_path, model, key="model", other_info=other_info
            )

        log_path = os.path.join(log_dir, "log.json")
        with open(log_path, 'w') as file:
            json.dump(history, file, indent=4)
        
    print(f'Best model achieved at epoch {best_epoch + 1} with accuracy: {best_metric * 100:.2f}%')

    # last model
    if args.save_last_model:
        last_model_path = os.path.join(log_dir, "model_last.pth")
        other_info = {
            "backbone": args.backbone,
            'pretrained': args.pretrained,
            'classifier': args.classifier,
        }
        utils_torch.save_checkpoint(
            last_model_path, model, key="model", other_info=other_info
        )
        print('Last checkpoint saved at', last_model_path)

    print('Start testing ...')    
    for test_csv_path, test_image_dir, test_att_label in zip(args.test, args.image_dir_test, args.att_label_test):
        dataset_name = test_image_dir.split("/")[2] 
        dataloader_test = get_test_dataloader(args, test_csv_path, test_image_dir, attribute_encoders, label_encoder, att_label=test_att_label)
        get_test_results(dataset_name, dataloader_test, model, history, log_path, att_label=test_att_label, save_log=True)

if __name__ == "__main__":
    since = time.time()
    args = setup_args()
    main(args)
    time_elapsed = time.time() - since
    print("Elapsed Time {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))