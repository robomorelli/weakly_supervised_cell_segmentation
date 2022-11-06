
import sys


sys.path.append('..')
from config import *
from dataset_loader.image_loader import *
from model.losses import *

clip_x_to0 = 1e-4

def SmashTo0(x):
    return 0*x

class InverseSquareRootLinearUnit(nn.Module):

    def __init__(self, min_value=5e-3):
        super(InverseSquareRootLinearUnit, self).__init__()
        self.min_value = min_value

    def forward(self, x):
        return 1. + self.min_value \
               + torch.where(torch.gt(x, 0), x, torch.div(x, torch.sqrt(1 + (x * x))))

class ClippedTanh(nn.Module):

    def __init__(self, min_value=5e-3):
        super(ClippedTanh, self).__init__()

    def forward(self, x):
        return 0.5 * (1 + 0.999 * torch.tanh(x))

class SmashTo0(nn.Module):

    def __init__(self):
        super(SmashTo0, self).__init__()

    def forward(self, x):
        return 0*x

class Dec1(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dec1, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class LinConstr(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinConstr, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class ConstrainedDec(nn.Linear):
    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return F.conv2d(input, self.weight.clamp(min=-1*10**6, max=1*10**6), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def load_data_train_eval(dataset='green', batch_size=16, validation_split=0.3, grayscale=False, num_workers=0,
                         shuffle_dataset=True, random_seed=42, ngpus=1,
                         ae=False, few_shot=False,  num_samples=100):


    transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                           T.ToTensor()
                           ])
    transform_l = T.Compose([
                           T.ToTensor()
                           ])
    if dataset == 'green':
        if few_shot:
            images_path = str(AugCropImagesFewShot) + '_{}'.format(num_samples)
            masks_path = str(AugCropMasksFewShot) + '_{}'.format(num_samples)
        else:
            images_path = aug_cropped_train_val_images
            masks_path = aug_cropped_train_val_masks
    elif dataset == 'blu':
        raise NotImplementedError

    cells_images = CellsLoader(images_path, masks_path, val_split=0.3, grayscale=grayscale, transform=transform, ae=ae)

    dataset_size = len(cells_images)
    print('dataset size is {}'.format(dataset_size))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices, )
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(cells_images, batch_size=batch_size * ngpus,
                              sampler=train_sampler, num_workers=num_workers)
    validation_loader = DataLoader(cells_images, batch_size=batch_size * ngpus,
                                   sampler=valid_sampler, num_workers=num_workers)

    return train_loader, validation_loader


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0.0001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                

def train_cycle(model, criterion, optimizer, train_loader, val_loader, device,
                save_model_path, added_path, scheduler, early_stopping, model_name, epochs=100):
    val_loss = 10**6
    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (image, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                y = target.to(device)
                x = image.to(device)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            ###############################################
            # eval mode for evaluation on validation dataset_loader
            ###############################################
            with torch.no_grad():
                model.eval()
                temp_val_loss = 0
                with tqdm(val_loader, unit="batch") as vepoch:
                    for i, (image, target) in enumerate(vepoch):
                        optimizer.zero_grad()

                        y = target.to(device)
                        x = image.to(device)
                        out = model(x)
                        loss = criterion(out, y)
                        temp_val_loss += loss
                        if i % 10 == 0:
                            print("VALIDATION Loss: {} batch {} on total of {}".format(loss.item(),
                                                                                       i, len(val_loader)))

                    temp_val_loss = temp_val_loss / len(val_loader)
                    print('validation_loss {}'.format(temp_val_loss))
                    scheduler.step(temp_val_loss)
                    if temp_val_loss < val_loss:
                        print('val_loss improved from {} to {}, saving model to {}' \
                          .format(val_loss, temp_val_loss, save_model_path.as_posix() + '/' + added_path + model_name))
                        print("saving model to {}".format(save_model_path.as_posix() + '/' + added_path + model_name))
                        path_posix = save_model_path.as_posix() + '/' + added_path + model_name
                        save_path = path_posix + '.h5'
                        val_loss = temp_val_loss
                        torch.save({'model_state_dict':model.state_dict(),
                                    'epoch':epoch,
                                    'lr':optimizer.param_groups[0]['lr'],
                                    'val_loss':val_loss}, save_path)

                    early_stopping(temp_val_loss)
                    if early_stopping.early_stop:
                        break


def vae_train_cycle(model, criterion, optimizer, train_loader, val_loader, device,
                save_model_path, added_path, scheduler, early_stopping, model_name, epochs=100, scale=10, kld_factor=0.6):
    val_loss = 10**6
    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (image, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                y = target.to(device)
                x = image.to(device)

                mu, sigma, segm, (mu_p, sigma_p) = model(x)
                segm_loss = criterion(segm, y)
                recon_loss, au, ne, au_1ch, ne_1ch = loss_VAE_rec(mu_p, sigma_p, x)

                KLD = KL_loss_forVAE(mu, sigma)
                loss = scale * recon_loss + kld_factor*KLD + scale*segm_loss

                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item(), nll = recon_loss.item(), KLD=KLD.item()*kld_factor, segm_loss=scale*segm_loss.item())
            ###############################################
            # eval mode for evaluation on validation dataset_loader
            ###############################################
            with torch.no_grad():
                model.eval()
                temp_val_loss = 0
                with tqdm(val_loader, unit="batch") as vepoch:
                    for i, (image, target) in enumerate(vepoch):
                        optimizer.zero_grad()

                        y = target.to(device)
                        x = image.to(device)

                        mu, sigma, segm, (mu_p, sigma_p) = model(x)
                        segm_loss = criterion(segm, y)
                        recon_loss, au, ne, au_1ch, ne_1ch = loss_VAE_rec(mu_p, sigma_p, x)

                        KLD = KL_loss_forVAE(mu, sigma)
                        loss = scale * recon_loss + kld_factor * KLD + scale * segm_loss

                        temp_val_loss += loss
                        if i % 10 == 0:
                            print("VALIDATION Loss: {} batch {} on total of {}".format(loss.item(),
                                                                                    i, len(val_loader)))

                    scheduler.step(temp_val_loss)
                    if temp_val_loss < val_loss:
                        print('val_loss improved from {} to {}, saving model to {}' \
                          .format(val_loss, temp_val_loss, save_model_path.as_posix() + '/' + added_path + model_name))
                        print("saving model to {}".format(save_model_path.as_posix() + '/' + added_path + model_name))
                        path_posix = save_model_path.as_posix() + '/' + added_path + model_name
                        save_path = path_posix + '.h5'
                        torch.save(model.state_dict(), save_path,
                                   epoch)
                        val_loss = temp_val_loss

                    early_stopping(temp_val_loss)
                    if early_stopping.early_stop:
                        break
