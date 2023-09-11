import matplotlib.pyplot as plt
import torch
import numpy as np

def show_decision_boundary(density_model, X_test, Y_test):
    x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
    y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    # Concatenate data_loaders to match input
    data = np.hstack((XX.ravel().reshape(-1, 1),
                      YY.ravel().reshape(-1, 1)))

    # Pass data_loaders to predict method
    db_prob = density_model(torch.from_numpy(data).to('cuda').float()).detach().cpu().numpy()

    clf = np.where(db_prob < 0.5, 0, 1)

    Z = clf.reshape(XX.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test,
                cmap=plt.cm.Accent)
    plt.show()

def show_cls_loss(losses):
    fig, axes = plt.subplots(1, 1, figsize=np.array([15, 6]))
    # loss_dict = {'pfl': []}
    axes.plot(losses)
    axes.set_title(f'BCELoss')
    plt.show()
    plt.close(fig)


import os
def save_loss(losses, name, train_dir):
    fig, axes = plt.subplots(1, 1, figsize=np.array([15, 6]))
    # loss_dict = {'pfl': []}
    axes.plot(losses)
    axes.set_title(f'{name}')
    plt.savefig(os.path.join(train_dir, f'{name}_fig.png'))
    plt.close(fig)

def show_train(Z_list, loss, epoch):

    fig1, axes1 = plt.subplots(1, 1, sharex=True, sharey=True)
    ax_iter = iter(np.array(axes1).ravel())
    # loss_dict = {'pfl': [], 'tpc': []}


    # loss_dict['pfl'].append(loss)
    # loss_dict['tpc'].append(loss_tpc.item())
    ax = next(ax_iter)
    data = torch.cat([Z.detach() for Z in Z_list], dim=0).to(torch.device('cpu'))

    for ii, Z in enumerate(Z_list):
        data = Z.detach().cpu().numpy()
        ax.scatter(*data.T, c=f'C{ii}', label=f'$Z_{ii + 1}$')
        ax.set_title(f'Epoch {epoch:3d}, Loss={loss:.5g}')
    ax.legend()
    plt.show()

    plt.close(fig1)

def show_test(X_list, models):
    for model in models:
        model.eval()

    Z_list = []
    for X, model in zip(X_list, models):
        z, _ = model(X)
        Z_list.append(z.detach().cpu().numpy())

    plt.figure()
    plt.scatter(*Z_list[0].T)
    plt.scatter(*Z_list[1].T)
    plt.show()

def show_inverse(transformed_x1, transformed_x2):
    plt.scatter(*transformed_x1.T)
    plt.scatter(*transformed_x2.T)
    plt.title('inverse')
    plt.show()


def load_part_of_model(model, path):
    params = torch.load(path)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try :
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                print(e)
                pass
    print('added %s of params:' % (added / float(len(model.state_dict().keys()))))

# def show_train_cifar()

def normalize_(z):  # B 1 H W ; [0,1] => [-1, 1]
    z_max = np.quantile(z.cpu().numpy(), 0.9)
    z_min = np.quantile(z.cpu().numpy(), 0.1)
    return ((z - z_min) / float(z_max - z_min)) * 2 - 1

def normalize_celeb(z):  # B 1 H W ; [0,1] => [-1, 1]
    z_max = z.max()
    z_min = z.min()
    return ((z - z_min) / float(z_max - z_min)) * 2 - 1
