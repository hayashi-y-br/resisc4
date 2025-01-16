import os
import sys

import hydra
import numpy as np
import torch
import torch.optim as optim
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score

from dataset import RESISC4
from early_stopping import EarlyStopping
from loss_function import CrossEntropyLoss
from models import ABMIL, AddMIL


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    sys.stdout = open('stdout.txt', 'w')
    sys.stderr = open('stderr.txt', 'w')
    os.makedirs('training')
    os.makedirs('validation')
    os.makedirs('test')

    group = HydraConfig.get().job.override_dirname.replace('dataset.', '').replace('model.', '').replace('settings.', '')
    name = f'{group},seed={cfg.seed}'
    run = wandb.init(
        project='RESISC4',
        name=name,
        group=group,
        config=OmegaConf.to_container(cfg)
    )

    cfg.use_cuda = cfg.use_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    if cfg.use_cuda:
        print(torch.cuda.get_device_name())
        torch.cuda.manual_seed(cfg.seed)

    print('Load Datasets')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        RESISC4(
            **cfg.dataset.settings,
            **cfg.dataset.train
        ),
        batch_size=cfg.settings.batch_size,
        shuffle=True,
        **loader_kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        RESISC4(
            **cfg.dataset.settings,
            **cfg.dataset.valid
        ),
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        RESISC4(
            **cfg.dataset.settings,
            **cfg.dataset.test
        ),
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )

    print('Init Model')
    if cfg.model.name == 'ABMIL':
        model = ABMIL(num_classes=cfg.model.num_classes)
    elif cfg.model.name in ['AddMIL_sigmoid', 'AddMIL_tanh']:
        model = AddMIL(num_classes=cfg.model.num_classes, activation=cfg.model.activation)

    if cfg.use_cuda:
        model.cuda()

    loss_fn = CrossEntropyLoss()
    early_stopping = EarlyStopping(min_delta=cfg.settings.min_delta, patience=cfg.settings.patience, model_path=cfg.model_path)
    optimizer = optim.Adam(model.parameters(), lr=cfg.settings.lr, weight_decay=cfg.settings.wd)

    print('Start Training')
    train_metrics_list = {f'training/{key}': [] for key in loss_fn.metrics}
    train_metrics_list['training/accuracy'] = []
    valid_metrics_list = {f'validation/{key}': [] for key in loss_fn.metrics}
    valid_metrics_list['validation/accuracy'] = []
    for epoch in range(1, cfg.settings.epochs + 1):
        # Training
        model.train()
        train_metrics = {key: 0. for key in train_metrics_list.keys()}
        for i, (X, y) in enumerate(train_loader):
            if cfg.use_cuda:
                X, y = X.cuda(), y.cuda()

            optimizer.zero_grad()
            output, output_dict = model(X)
            loss, metrics = loss_fn(output, y, epoch)
            loss.backward()
            optimizer.step()

            y_hat = output_dict['y_hat']
            train_metrics['training/accuracy'] += y_hat.eq(y).detach().cpu().sum(dtype=float)
            for key, value in metrics.items():
                train_metrics[f'training/{key}'] += value

        for key, value in train_metrics.items():
            value /= len(train_loader.dataset)
            train_metrics[key] = value
            train_metrics_list[key].append(value)
        wandb.log(train_metrics, commit=False)
        print('Epoch: {:3d}, Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(epoch, train_metrics['training/total_loss'], train_metrics['training/accuracy']), end=', ')

        # Validation
        model.eval()
        valid_metrics = {key: 0. for key in valid_metrics_list.keys()}
        with torch.no_grad():
            for i, (X, y) in enumerate(valid_loader):
                if cfg.use_cuda:
                    X, y = X.cuda(), y.cuda()

                output, output_dict = model(X)
                loss, metrics = loss_fn(output, y, epoch)

                y_hat = output_dict['y_hat']
                valid_metrics[f'validation/accuracy'] += y_hat.eq(y).detach().cpu().sum(dtype=float)
                for key, value in metrics.items():
                    valid_metrics[f'validation/{key}'] += value

        for key, value in valid_metrics.items():
            value /= len(valid_loader.dataset)
            valid_metrics[key] = value
            valid_metrics_list[key].append(value)
        wandb.log(valid_metrics)
        print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(valid_metrics['validation/total_loss'], valid_metrics['validation/accuracy']))
        if early_stopping(valid_metrics['validation/total_loss'], model):
            break
    wandb.save(cfg.model_path)
    for key, value in train_metrics_list.items():
        value = np.array(value)
        np.savetxt(f'{key}.csv', value, delimiter=',')
    for key, value in valid_metrics_list.items():
        value = np.array(value)
        np.savetxt(f'{key}.csv', value, delimiter=',')

    print('Start Testing')
    y_list = []
    y_hat_list = []
    model.load_state_dict(torch.load(cfg.model_path, weights_only=True))
    model.eval()
    test_metrics = {f'test/{key}': 0. for key in loss_fn.metrics}
    test_metrics['test/accuracy'] = 0.
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if cfg.use_cuda:
                X, y = X.cuda(), y.cuda()

            output, output_dict = model(X)
            loss, metrics= loss_fn(output, y)

            y_hat = output_dict['y_hat']
            test_metrics['test/accuracy'] += y_hat.eq(y).detach().cpu().sum(dtype=float)
            for key, value in metrics.items():
                test_metrics[f'test/{key}'] += value

            y = y.detach().cpu()[0]
            y_hat = y_hat.detach().cpu()[0]
            y_list.append(y)
            y_hat_list.append(y_hat)

    for key, value in test_metrics.items():
        value /= len(test_loader.dataset)
        test_metrics[key] = value
        wandb.summary[key] = value
        np.savetxt(f'{key}.csv', [value], delimiter=',')
    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_metrics['test/total_loss'], test_metrics['test/accuracy']))

    y = np.array(y_list)
    y_hat = np.array(y_hat_list)
    metric = f1_score(y, y_hat, average='macro')
    np.savetxt('y_true.csv', y, delimiter=',')
    np.savetxt('y_pred.csv', y_hat, delimiter=',')
    np.savetxt('f1_score.csv', [metric], delimiter=',')

    wandb.save('y_true.csv')
    wandb.save('y_pred.csv')
    wandb.summary['f1_score'] = metric
    wandb.finish()

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == '__main__':
    main()