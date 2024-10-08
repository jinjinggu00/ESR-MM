"""Training program for early fusion of PAINet"""
# coding=utf-8
import random
import fitlog

from backbone_select import Backbone
from protonet import ProtoNet
from prototypical_batch_sampler import PrototypicalBatchSampler
from nturgbd_dataset import NTU_RGBD_Dataset
from parser_util import get_parser
from utils import load_data, get_para_num, setup_seed, getAvaliableDevice
from tqdm import tqdm
import numpy as np
import torch
import pickle
import os
import time
import gl
import warnings
from utils import *

fitlog.debug()
fitlog.set_log_dir('logs')
fitlog.commit(__file__, 'ESR-MM with painet early fusion')
fitlog.add_hyper_in_file(__file__)  # record your hyperparameters

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def init_seed(opt):
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, data_list, mode):
    debug = False
    dataset = NTU_RGBD_Dataset(mode=mode, data_list=data_list, debug=debug, extract_frame=opt.extract_frame,
                               modal=opt.modal, process=opt.process)
    n_classes = len(np.unique(dataset.label))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen classes_per_it. Decrease the ' +
                         'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
        iters = opt.train_iterations
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val
        iters = opt.test_iterations

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iters)


def init_dataloader(opt, data_list, mode):
    dataset = init_dataset(opt, data_list, mode)
    sampler = init_sampler(opt, dataset.label, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=6,
                                             pin_memory=False)
    return dataloader


def init_protonet(opt):
    model = ProtoNet(opt).to(gl.device)
    if opt.model == 1:
        model_path = os.path.join(opt.experiment_root, 'best_model.pth')
        # print('model_path', model_path)
        model.load_state_dict(torch.load(model_path))
    # print(get_para_num(model))
    return model


def init_optim(opt, all_params):
    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(all_params, lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif opt.optim == 'adam':
        optimizer = torch.optim.Adam(params=all_params, lr=opt.learning_rate, weight_decay=5e-4)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(params=all_params, lr=opt.learning_rate, weight_decay=5e-4)
    elif opt.optim == 'radam':
        from torch_optimizer import RAdam
        optimizer = RAdam(all_params, lr=opt.learning_rate, weight_decay=5e-4)
    elif opt.optim == 'lookahead':
        from torch_optimizer import Lookahead
        base_optimizer = torch.optim.AdamW(params=all_params, lr=opt.learning_rate, weight_decay=5e-4)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    else:
        raise ValueError('Invalid optimizer')
    return optimizer


def init_lr_scheduler(opt, optim, train_loader):
    if opt.lr_flag == 'reduceLR':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
                                                                  verbose=True, min_lr=1e-5)
    elif opt.lr_flag == 'stepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=opt.lr_scheduler_gamma,
                                                       step_size=opt.lr_scheduler_step)
    elif opt.lr_flag == 'one':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim,
            max_lr=opt.learning_rate * 10,
            steps_per_epoch=len(train_loader),
            epochs=opt.epochs,
            pct_start=0.3
        )
    elif opt.lr_flag == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=opt.epochs)
    elif opt.lr_flag == 'cosine_warm':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
    else:
        raise ValueError('Invalid lr_scheduler')
    return lr_scheduler


def init_model(opt, model):
    if opt.process == 1:
        if opt.modal == 2:
            model1, _, _ = Backbone(opt.dataset, 'blcok3', 3)
            model2, _, _ = Backbone(opt.dataset, 'blcok3', 3)
            all_params = (list(model.parameters()) +
                          list(model1.parameters()) +
                          list(model2.parameters()))
            return model1.to(gl.device), model2.to(gl.device), all_params
        elif opt.modal == 3:
            model1, _, _ = Backbone(opt.dataset, 'blcok3', 3)
            model2, _, _ = Backbone(opt.dataset, 'blcok3', 3)
            model3, _, _ = Backbone(opt.dataset, 'blcok3', 3)
            all_params = (list(model.parameters()) +
                          list(model1.parameters()) +
                          list(model2.parameters()) +
                          list(model3.parameters()))
            return model1.to(gl.device), model2.to(gl.device), model3.to(gl.device), all_params
        elif opt.modal == 4:
            model1, _, _ = Backbone(opt.dataset, 'blcok3', 3)
            model2, _, _ = Backbone(opt.dataset, 'blcok3', 3)
            model3, _, _ = Backbone(opt.dataset, 'blcok3', 3)
            model4, _, _ = Backbone(opt.dataset, 'blcok3', 3)
            all_params = (list(model.parameters()) +
                          list(model1.parameters()) +
                          list(model2.parameters()) +
                          list(model3.parameters()) +
                          list(model4.parameters()))
            return model1.to(gl.device), model2.to(gl.device), model3.to(gl.device), model4.to(gl.device), all_params
        else:
            ValueError('Unknown modal, you should choose 2, 3 or 4.')


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None, test_dataloader=None, start_epoch=0,
          beacc=0, model1=None, model2=None, model3=None, model4=None):
    scaler = torch.cuda.amp.GradScaler()
    import json
    with open(os.path.join(opt.experiment_root, 'opt.json'), 'w') as f:
        j = vars(opt)
        json.dump(j, f)
        f.write('\n')

    if val_dataloader is None:
        best_state = None

    best_acc = beacc
    best_epoch = 0
    last_acc = 0
    acc_reduce_num = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    last_opmodel_path = os.path.join(opt.experiment_root, 'last_opmodel.pth')
    trace_file = os.path.join(opt.experiment_root, 'trace.txt')

    patience = 0

    for epoch in range(start_epoch, opt.epochs):
        gl.epoch = epoch
        gl.iter = 0
        print(f'=== Epoch:{epoch}===')
        tr_iter = iter(tr_dataloader)
        model.train()
        train_acc = []
        reg_loss = []
        train_loss = []
        pca_loss = []
        for batch in tqdm(tr_iter):
            # for batch in tr_iter:
            optim.zero_grad()
            gl.mod = 'train'
            if opt.process == 1:
                if opt.modal == 2:
                    x, x1, y = batch
                    if type(y) is list:
                        y = [int(label) for label in y]
                        y = torch.tensor(y)
                    x, x1, y = x.to(gl.device).float(), x1.to(gl.device).float(), y.to(gl.device)
                    o1 = model1(x)
                    o2 = model2(x1)
                    x = torch.cat([o1, o2], dim=1)
                elif opt.modal == 3:
                    x, x1, x2, y = batch
                    if type(y) is list:
                        y = [int(label) for label in y]
                        y = torch.tensor(y)
                    x, x1, x2, y = x.to(gl.device).float(), x1.to(gl.device).float(), x2.to(gl.device).float(), y.to(
                        gl.device)
                    o1 = model1(x)
                    o2 = model2(x1)
                    o3 = model3(x2)
                    x = torch.cat([o1, o2, o3], dim=1)
                elif opt.modal == 4:
                    x, x1, x2, x3, y = batch
                    if type(y) is list:
                        y = [int(label) for label in y]
                        y = torch.tensor(y)
                    x, x1, x2, x3, y = x.to(gl.device).float(), x1.to(gl.device).float(), x2.to(
                        gl.device).float(), x3.to(
                        gl.device).float(), y.to(gl.device)
                    o1 = model1(x)
                    o2 = model2(x1)
                    o3 = model3(x2)
                    o4 = model4(x3)
                    x = torch.cat([o1, o2, o3, o4], dim=1)
                else:
                    ValueError('Unknown modal, you should choose 2, 3 or 4.')
            else:
                x, y = batch
                if type(y) is list:
                    y = [int(label) for label in y]
                    y = torch.tensor(y)
                x, y = x.to(gl.device).float(), y.to(gl.device)
            # x torch.Size([55, 3, 30, 25, 2])
            model_output = model(x)

            # model_output torch.Size([55, 256, 8, 25])

            loss, acc, reg, pca = model.train_mode(model_output, y, opt.num_support_tr)

            train_loss.append(loss.item())
            train_acc.append(acc.item())
            reg_loss.append(reg.item())
            pca_loss.append(pca.item())

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            scaler.step(optim)
            scaler.update()
        avg_loss = np.mean(train_loss)
        avg_reg = np.mean(reg_loss)
        avg_acc = np.mean(train_acc)
        avg_pca = np.mean(pca_loss)

        string = (f'train loss:{avg_loss}, classfier loss:{avg_loss - avg_reg - avg_pca}, '
                  f'reg loss:{avg_reg}, pca loss:{avg_pca}, train Acc: {avg_acc}')

        fitlog.add_loss(step=epoch, name='train_loss', value=avg_loss.item())
        fitlog.add_loss(step=epoch, name='classfier_loss', value=(avg_loss - avg_reg).item())
        fitlog.add_loss(step=epoch, name='reg_loss', value=avg_reg.item())
        fitlog.add_loss(step=epoch, name='pca_loss', value=avg_pca.item())
        fitlog.add_metric(step=epoch, name='train Acc', value=avg_acc.item())
        if opt.lr_flag == 'reduceLR':
            lr_scheduler.step(avg_loss)
        elif opt.lr_flag in ['stepLR', 'one', 'cosine', 'cosine_warm']:
            lr_scheduler.step()
        else:
            raise ValueError(f"Unexpected lr_flag value: {opt.lr_flag}")

        lr = optim.state_dict()['param_groups'][0]['lr']

        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)

        model.eval()

        val_acc = []
        with torch.no_grad():
            for batch in tqdm(val_iter):
                if opt.process == 1:
                    if opt.modal == 2:
                        x, x1, y = batch
                        if type(y) is list:
                            y = [int(label) for label in y]
                            y = torch.tensor(y)
                        x, x1, y = x.to(gl.device).float(), x1.to(gl.device).float(), y.to(gl.device)
                        o1 = model1(x)
                        o2 = model2(x1)
                        x = torch.cat([o1, o2], dim=1)
                    elif opt.modal == 3:
                        x, x1, x2, y = batch
                        if type(y) is list:
                            y = [int(label) for label in y]
                            y = torch.tensor(y)
                        x, x1, x2, y = x.to(gl.device).float(), x1.to(gl.device).float(), x2.to(
                            gl.device).float(), y.to(
                            gl.device)
                        o1 = model1(x)
                        o2 = model2(x1)
                        o3 = model3(x2)
                        x = torch.cat([o1, o2, o3], dim=1)
                    elif opt.modal == 4:
                        x, x1, x2, x3, y = batch
                        if type(y) is list:
                            y = [int(label) for label in y]
                            y = torch.tensor(y)
                        x, x1, x2, x3, y = x.to(gl.device).float(), x1.to(gl.device).float(), x2.to(
                            gl.device).float(), x3.to(
                            gl.device).float(), y.to(gl.device)
                        o1 = model1(x)
                        o2 = model2(x1)
                        o3 = model3(x2)
                        o4 = model4(x3)
                        x = torch.cat([o1, o2, o3, o4], dim=1)
                    else:
                        ValueError('Unknown modal, you should choose 2, 3 or 4.')
                else:
                    x, y = batch
                    if type(y) is list:
                        y = [int(label) for label in y]
                        y = torch.tensor(y)
                    x, y = x.to(gl.device).float(), y.to(gl.device)
                gl.mod = 'val'
                model_output = model(x)
                acc = model.evaluate(model_output, target=y, n_support=opt.num_support_val)

                val_acc.append(acc.item())

        avg_acc = np.mean(val_acc)

        # if acc reduce 10 times, break
        if last_acc == 0:
            last_acc = avg_acc
        else:
            if last_acc >= avg_acc:
                acc_reduce_num += 1
            else:
                acc_reduce_num = 0
            last_acc = avg_acc
        if acc_reduce_num >= 10:
            print('acc already reduce more than 10 times!!  end training...')
            fitlog.add_other(name='end_epoch', value=epoch)
            break

        postfix = ' (Best)' if avg_acc >= best_acc else f' (Best: {best_acc})'
        # string_val = f'val loss: {avg_loss}, val acc: {avg_acc}{postfix} lr:{lr}'
        string_val = f'val acc: {avg_acc}{postfix} lr:{lr}'

        fitlog.add_metric(step=epoch, name='val Acc', value=avg_acc)
        print(string + '\t' + string_val)
        with open(trace_file, 'a') as f:
            f.write(string + '\t' + string_val)
            f.write('\n')

        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            patience = 0
            best_acc = avg_acc
            best_epoch = epoch
            best_state = model.state_dict()
            if opt.modal == 2:
                bs1 = model1.state_dict()
                bs2 = model2.state_dict()
                bs3 = None
                bs4 = None
            elif opt.modal == 3:
                bs1 = model1.state_dict()
                bs2 = model2.state_dict()
                bs3 = model3.state_dict()
                bs4 = None
            elif opt.modal == 4:
                bs1 = model1.state_dict()
                bs2 = model2.state_dict()
                bs3 = model3.state_dict()
                bs4 = model4.state_dict()
        else:
            patience += 1

        if patience > 40:
            fitlog.add_other(name='end_epoch', value=epoch)
            break

        if epoch % 10 == 0:
            state = {'model': model.state_dict(), 'optimizer': optim.state_dict(), 'epoch': epoch,
                     'best_acc': best_acc}
            torch.save(state, last_opmodel_path)

    torch.save(model.state_dict(), last_model_path)
    state = {'model': model.state_dict(), 'optimizer': optim.state_dict(), 'epoch': epoch,
             'best_acc': best_acc}
    torch.save(state, last_opmodel_path)
    fitlog.add_other(name='end_epoch', value=epoch)
    fitlog.add_other(name='best_trepoch', value=best_epoch)
    fitlog.add_best_metric(name='best_tracc', value=best_acc)
    return best_state, best_acc, bs1, bs2, bs3, bs4


def test(opt, test_dataloader, model, model1=None, model2=None, model3=None, model4=None):
    '''
    Test the model trained with the prototypical learning algorithm
    '''

    print('testing model...')
    avg_acc = list()
    trace_file = os.path.join(opt.experiment_root, 'test.txt')

    with torch.no_grad():
        for epoch in range(10):
            print(f'=== Epoch:{epoch}===')
            model.eval()
            gl.epoch = epoch
            test_iter = iter(test_dataloader)
            for batch in test_iter:
                if opt.process == 1:
                    if opt.modal == 2:
                        x, x1, y = batch
                        if type(y) is list:
                            y = [int(label) for label in y]
                            y = torch.tensor(y)
                        x, x1, y = x.to(gl.device).float(), x1.to(gl.device).float(), y.to(gl.device)
                        o1 = model1(x)
                        o2 = model2(x1)
                        x = torch.cat([o1, o2], dim=1)
                    elif opt.modal == 3:
                        x, x1, x2, y = batch
                        if type(y) is list:
                            y = [int(label) for label in y]
                            y = torch.tensor(y)
                        x, x1, x2, y = x.to(gl.device).float(), x1.to(gl.device).float(), x2.to(
                            gl.device).float(), y.to(
                            gl.device)
                        o1 = model1(x)
                        o2 = model2(x1)
                        o3 = model3(x2)
                        x = torch.cat([o1, o2, o3], dim=1)
                    elif opt.modal == 4:
                        x, x1, x2, x3, y = batch
                        if type(y) is list:
                            y = [int(label) for label in y]
                            y = torch.tensor(y)
                        x, x1, x2, x3, y = x.to(gl.device).float(), x1.to(gl.device).float(), x2.to(
                            gl.device).float(), x3.to(
                            gl.device).float(), y.to(gl.device)
                        o1 = model1(x)
                        o2 = model2(x1)
                        o3 = model3(x2)
                        o4 = model4(x3)
                        x = torch.cat([o1, o2, o3, o4], dim=1)
                    else:
                        ValueError('Unknown modal, you should choose 2, 3 or 4.')
                else:
                    x, y = batch
                    if type(y) is list:
                        y = [int(label) for label in y]
                        y = torch.tensor(y)
                    x, y = x.to(gl.device).float(), y.to(gl.device)
                model_output = model(x)
                # _, acc, _, _ = model.train_mode(model_output, target=y, n_support=opt.num_support_val)
                acc = model.evaluate(model_output, target=y, n_support=opt.num_support_val)
                avg_acc.append(acc.item())
            fitlog.add_metric(step=epoch, name='test Acc', value=(np.mean(avg_acc)).item())
            # print('test avg_acc', np.mean(avg_acc))

    avg_acc = np.mean(avg_acc)
    fitlog.add_best_metric(name='Acc', value=avg_acc.item())
    with open(trace_file, 'a') as f:
        f.write(f'dataset:({opt.dataset})__back:({opt.backbone})__seed{opt.manual_seed}__'
                f'reg{opt.reg_rate}__metric{opt.metric}__hal{opt.AA}__pca{opt.pca}\n')
        f.write(f'test acc: {avg_acc}\n')
    print(f'Test Acc: {avg_acc}')

    return avg_acc


def main():
    """
    Initialize everything and train
    """
    options = get_parser().parse_args()
    options.experiment_root = os.path.join(
        options.experiment_root,
        f"seed_{options.manual_seed}",
        f"_dataset{options.dataset}",
        f"_back{options.backbone}",
        f"_metric{options.metric}_reg{options.reg_rate}_hal{options.AA}_pca{options.pca}"
    )

    options.cuda = True
    options.device = str(0)

    if options.debug == 1:
        gl.debug = True

    gl.device = 'cuda:{}'.format(options.device) if torch.cuda.is_available() and options.cuda else 'cpu'

    gl.gamma = options.gamma
    options.experiment_root = "log/" + options.experiment_root
    gl.experiment_root = options.experiment_root
    gl.reg_rate = options.reg_rate
    gl.threshold = options.thred
    gl.backbone = options.backbone
    gl.dataset = options.dataset
    gl.AA = options.AA
    gl.pca = options.pca
    gl.metric = options.metric
    gl.modal = options.modal
    gl.vel = options.vel
    gl.bone = options.bone
    print('Dataset:', gl.dataset)
    print('Backbone:', gl.backbone)
    print('Metric:', gl.metric)
    print(f'{options.classes_per_it_tr}_way  {options.num_support_tr}_shot')
    print('Train epochs:', options.epochs)
    fitlog.add_progress(total_steps=options.epochs)
    fitlog.add_hyper(name='dataset', value=gl.dataset)
    fitlog.add_hyper(name='backbone', value=gl.backbone)
    fitlog.add_hyper(name='reg_rate', value=gl.reg_rate)
    fitlog.add_hyper(name='HAL', value=gl.AA)
    fitlog.add_hyper(name='pca', value=gl.pca)
    fitlog.add_hyper(name='metric', value=options.metric)
    fitlog.add_hyper(name='way', value=options.classes_per_it_tr)
    fitlog.add_hyper(name='shot', value=options.num_support_tr)
    fitlog.add_hyper(name='modal', value=options.modal)

    if not os.path.exists(gl.experiment_root):
        os.makedirs(gl.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if options.process == 1:
        if options.modal == 2:
            gl.num_chanels = 128
        elif options.modal == 3:
            gl.num_chanels = 192
        elif options.modal == 4:
            gl.num_chanels = 256
        else:
            ValueError('Unknown modal, you should choose 2, 3 or 4.')

    init_seed(options)
    setup_seed(options.manual_seed)
    fitlog.add_hyper(name='seed', value=options.manual_seed)
    data_list = []

    tr_dataloader = init_dataloader(options, data_list, 'train')
    val_dataloader = init_dataloader(options, data_list, 'val')
    test_dataloader = init_dataloader(options, data_list, 'test')

    model = init_protonet(options)
    all_params = model.parameters()
    model1 = None
    model2 = None
    model3 = None
    model4 = None

    if options.process == 1:
        if options.modal == 2:
            model1, model2, all_params = init_model(options, model)
        elif options.modal == 3:
            model1, model2, model3, all_params = init_model(options, model)
        elif options.modal == 4:
            model1, model2, model3, model4, all_params = init_model(options, model)
        else:
            ValueError('Unknown modal, you should choose 2, 3 or 4.')

    optim = init_optim(options, all_params)
    lr_scheduler = init_lr_scheduler(options, optim, tr_dataloader)

    if options.mode == 'train':
        epoch = 0
        bsacc = 0
        if options.contin == 1:
            "Load saved weights and continue training"
            model_path = os.path.join(options.experiment_root, 'last_opmodel.pth')
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model'])
            optim.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            bsacc = checkpoint['best_acc']
        res = train(opt=options,
                    tr_dataloader=tr_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler,
                    start_epoch=epoch,
                    beacc=bsacc,
                    model1=model1,
                    model2=model2,
                    model3=model3,
                    model4=model4)
        best_state, best_acc, bs1, bs2, bs3, bs4 = res

        model.load_state_dict(best_state)
        model_path = os.path.join(options.experiment_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        if options.modal == 2:
            model1.load_state_dict(bs1)
            model2.load_state_dict(bs2)
        elif options.modal == 3:
            model1.load_state_dict(bs1)
            model2.load_state_dict(bs2)
            model3.load_state_dict(bs3)
        elif options.modal == 4:
            model1.load_state_dict(bs1)
            model2.load_state_dict(bs2)
            model3.load_state_dict(bs3)
            model4.load_state_dict(bs4)

        print('Testing with best model..')
        test(opt=options,
             test_dataloader=test_dataloader,
             model=model, model1=model1, model2=model2, model3=model3, model4=model4)
    elif options.mode == 'test':  # -mode test
        model_path = os.path.join(options.experiment_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        print('Testing with best model..')
        test(opt=options,
             test_dataloader=test_dataloader,
             model=model, model1=model1, model2=model2, model3=model3, model4=model4)


if __name__ == '__main__':
    device = torch.device("cuda")
    main()
    fitlog.finish()
