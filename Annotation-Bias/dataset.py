from torchvision import datasets, transforms
from torch.utils.data import random_split
from PIL import Image
import torch
import numpy as np
import os
from math import inf
from scipy import stats
from torch.nn import functional as F


MNIST_MEAN = [0.1307]
MNIST_STD = [0.3081]
CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]
CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR10_HUMAN_NOISE_PATH = './data/CIFAR-10_human.pt'
CIFAR100_HUMAN_NOISE_PATH = './data/CIFAR-100_human.pt'

def get_sym_T(eta, num_classes):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    diag_mask = np.eye(num_classes)
    rest_mask = 1 - diag_mask
    
    T = diag_mask * (1 - eta) \
        + rest_mask * eta / (num_classes - 1)
    
    return T

def get_asym_T_mnist(eta, single_edge=False):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # 7 -> 1
    T[7, 7], T[7, 1] = 1. - eta, eta
    # 2 -> 7
    T[2, 2], T[2, 7] = 1. - eta, eta
    # 5 <-> 6
    T[5, 5], T[5, 6] = 1. - eta, eta
    if not single_edge:
        T[6, 6], T[6, 5] = 1. - eta, eta
    # 3 -> 8
    T[3, 3], T[3, 8] = 1. - eta, eta
    
    return T

def get_asym_T_cifar10(eta, single_edge=False):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # truck -> automobile (9 -> 1)
    T[9, 9], T[9, 1] = 1. - eta, eta
    # bird -> airplane (2 -> 0)
    T[2, 2], T[2, 0] = 1. - eta, eta
    # cat <-> dog (3 <-> 5)
    T[3, 3], T[3, 5] = 1. - eta, eta
    if not single_edge:
        T[5, 5], T[5, 3] = 1. - eta, eta
    # deer -> horse (4 -> 7)
    T[4, 4], T[4, 7] = 1. - eta, eta
    
    return T

def get_column_T_cifar10(eta):
    T = np.zeros((10,10))
    T[0][0],T[0][3],T[0][5] = 1.0 - eta, eta/2.0, eta/2.0
    T[1][1],T[1][3],T[1][5] = 1.0 - eta, eta/2.0, eta/2.0
    T[2][2],T[2][3],T[2][5] = 1.0 - eta, eta/2.0, eta/2.0
    T[3][3],T[3][5] = 1.0 - eta + 0.2, eta - 0.2
    T[4][4],T[4][3],T[4][5] = 1.0 - eta, eta/2.0, eta/2.0
    T[5][5],T[5][3] = 1.0 - eta + 0.2, eta - 0.2
    T[6][6],T[6][3],T[6][5] = 1.0 - eta, eta/2.0, eta/2.0
    T[7][7],T[7][3],T[7][5] = 1.0 - eta, eta/2.0, eta/2.0
    T[8][8],T[8][3],T[8][5] = 1.0 - eta, eta/2.0, eta/2.0
    T[9][9],T[9][3],T[9][5] = 1.0 - eta, eta/2.0, eta/2.0
    return T

def get_triangular_T_cifar10(eta):
    T = np.zeros((10,10))
    T[0][0],T[0][1] = 1.0 - eta + 0.2, eta - 0.2
    T[1][0],T[1][1],T[1][2] = eta/2.0, 1.0 - eta, eta/2.0
    T[2][1],T[2][2],T[2][3] = eta/2.0, 1.0 - eta, eta/2.0
    T[3][2],T[3][3],T[3][4] = eta/2.0, 1.0 - eta, eta/2.0
    T[4][3],T[4][4],T[4][5] = eta/2.0, 1.0 - eta, eta/2.0
    T[5][4],T[5][5],T[5][6] = eta/2.0, 1.0 - eta, eta/2.0
    T[6][5],T[6][6],T[6][7] = eta/2.0, 1.0 - eta, eta/2.0
    T[7][6],T[7][7],T[7][8] = eta/2.0, 1.0 - eta, eta/2.0
    T[8][7],T[8][8],T[8][9] = eta/2.0, 1.0 - eta, eta/2.0
    T[9][8],T[9][9] = eta - 0.2, 1.0 - eta + 0.2
  
    return T
    
def get_asym_T_cifar100(eta, single_edge=False):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 100
    num_superclasses = 20
    num_subclasses = 5

    T = np.eye(num_classes)

    for i in np.arange(num_superclasses):
        # build T for one superclass
        T_superclass = (1. - eta) * np.eye(num_subclasses)
        for j in np.arange(num_subclasses - 1):
            T_superclass[j, j + 1] = eta
        if not single_edge:
            T_superclass[num_subclasses - 1, 0] = eta
        
        init, end = i * num_subclasses, (i + 1) * num_subclasses
        T[init:end, init:end] = T_superclass

    return T

def create_noisy_labels(labels, trans_matrix):
    '''
    create noisy labels from labels and noisy matrix
    '''
    
    if trans_matrix is None:
        raise ValueError('Noisy matrix is None')
    
    num_trans_matrix = trans_matrix.copy()
    labels = labels.copy()
    
    num_classes = len(trans_matrix)
    class_idx = [np.where(np.array(labels) == i)[0]
                 for i in range(num_classes)]
    num_samples_class = [len(class_idx[idx])
                         for idx in range(num_classes)]
    for real_label in range(num_classes):
        for trans_label in range(num_classes):
            num_trans_matrix[real_label][trans_label] = \
                trans_matrix[real_label][trans_label] * num_samples_class[real_label]
    num_trans_matrix = num_trans_matrix.astype(int)

    for real_label in range(num_classes):
        for trans_label in range(num_classes):

            if real_label == trans_label:
                continue

            num_trans = num_trans_matrix[real_label][trans_label]
            if num_trans == 0:
                continue

            trans_samples_idx = np.random.choice(class_idx[real_label],
                                                 num_trans,
                                                 replace=False)
            class_idx[real_label] = np.setdiff1d(class_idx[real_label],
                                                 trans_samples_idx)
            for idx in trans_samples_idx:
                labels[idx] = trans_label
    
    return labels

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed): 
    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 
    print("building dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    to_tensor = transforms.ToTensor()

    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = to_tensor(x).cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)

def load_saved_labels(file_path, dataset_size):
    """
    Load saved predictions from file and reorder them by index.
    The file format is: index\tlabel
    Returns a numpy array of labels ordered by index.
    """
    labels_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            idx, label = line.strip().split('\t')
            labels_dict[int(idx)] = int(label)
    
    # Create ordered array
    ordered_labels = np.zeros(dataset_size, dtype=np.int64)
    for idx in range(dataset_size):
        if idx not in labels_dict:
            raise ValueError(f"Missing index {idx} in saved labels file")
        ordered_labels[idx] = labels_dict[idx]
    
    return ordered_labels

class MyMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None, return_index=False):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        self.return_index = return_index
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets.numpy(), trans_matrix)
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.return_index:
            return img, target, index
        return img, target

class MyCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None, noisy_targets=None, return_index=False):
        super().__init__(root, train, transform, target_transform, download)
        
        # Store clean targets before applying noise
        self.clean_targets = self.targets.copy() if isinstance(self.targets, list) else list(self.targets)
        
        self.trans_matrix = trans_matrix
        self.return_index = return_index
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets, trans_matrix)
        if noisy_targets is not None:
            self.targets = noisy_targets.tolist()
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Return image, noisy target, and optionally index
        if self.return_index:
            #need the index for when saving the labels for retraining after label correction
            return img, target, index
        return img, target

class MyCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None, noisy_targets=None, return_index=False):
        super().__init__(root, train, transform, target_transform, download)
        
        # Store clean targets before applying noise
        self.clean_targets = self.targets.copy() if isinstance(self.targets, list) else list(self.targets)
        
        self.trans_matrix = trans_matrix
        self.return_index = return_index
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets, trans_matrix)
        if noisy_targets is not None:
            self.targets = noisy_targets.tolist()
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Return image, noisy target, and optionally index
        if self.return_index:
            #need the index for when saving the labels for retraining after label correction
            return img, target, index
        return img, target

    
def mnist(root, noise_type, noise_rate, tuning=False):
    if noise_type == 'sym':
        T = get_sym_T(noise_rate, 10)
    elif 'asym' in noise_type:
        T = get_asym_T_mnist(noise_rate, noise_type=='asym_single')
    else:
        raise ValueError('Wrong noise type! Must be sym or asym')
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    
    if tuning:
        train_dataset = MyMNIST(root=root,
                                train=True,
                                transform=train_transform,
                                trans_matrix=T)
        num_train = int(len(train_dataset) * 0.9)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        train_dataset.trans_matrix = T
        
        eval_dataset = MyMNIST(root=root,
                               train=True,
                               transform=eval_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))

    else:
        train_dataset = MyMNIST(root=root,
                                train=True,
                                transform=train_transform,
                                trans_matrix=T)
        
        eval_dataset = MyMNIST(root=root,
                               train=False,
                               transform=eval_transform)
    
    return train_dataset, eval_dataset

def cifar10(root, noise_type, noise_rate, seed, tuning=False, saved_labels_file=None):
    # Initialize variables
    noisy_targets = None
    T = None
    
    # If saved labels file is provided, load it directly
    if saved_labels_file is not None and os.path.exists(saved_labels_file):
        # We'll load this later after we know it's not tuning mode
        pass
    elif noise_type == 'sym':
        T = get_sym_T(noise_rate, 10)
    elif 'asym' in noise_type:
        T = get_asym_T_cifar10(noise_rate, noise_type=='asym_single')
    elif noise_type == 'human':
        noisy_targets = torch.load(CIFAR10_HUMAN_NOISE_PATH,weights_only=False)['worse_label']
    elif noise_type == 'instance':
        #use instance-dependent noise
        #load cifar10 training set without any noise
        base_dataset = datasets.CIFAR10(root=root, train=True, download=True)
        base_data = base_dataset.data
        base_labels = np.array(base_dataset.targets)
        #convert to torch tensor
        base_data = torch.from_numpy(base_data).float()
        base_labels = torch.from_numpy(base_labels)
        feature_size = 3*32*32
        noisy_targets = get_instance_noisy_label(noise_rate, base_dataset, base_labels, 10, feature_size, 0.1, seed)
    elif noise_type == 'column':
        T = get_column_T_cifar10(noise_rate)
    elif noise_type == 'tri':
        T = get_triangular_T_cifar10(noise_rate)
    else:
        raise ValueError('Wrong noise type! Must be sym or asym')
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    
    if tuning:
        train_dataset = MyCIFAR10(root=root,
                                  train=True,
                                  transform=train_transform,
                                  trans_matrix=T)
        num_train = int(len(train_dataset) * 0.9)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        train_dataset.trans_matrix = T

        eval_dataset = MyCIFAR10(root=root,
                                 train=True,
                                 transform=eval_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))

    else:#use human noise only for validation
        # Load saved labels if provided
        if saved_labels_file is not None and os.path.exists(saved_labels_file):
            temp_dataset = datasets.CIFAR10(root=root, train=True, download=True)
            dataset_size = len(temp_dataset)
            noisy_targets = load_saved_labels(saved_labels_file, dataset_size)
            print(f"Loaded {len(noisy_targets)} labels from {saved_labels_file}")
            train_dataset = MyCIFAR10(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=None,
                                    noisy_targets=noisy_targets)
            # Set identity matrix for trans_matrix, needed for selalphha* methods
            train_dataset.trans_matrix = np.eye(10)
        elif noise_type not in ['human','instance']:
            train_dataset = MyCIFAR10(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=T)
        else:
            # For human/instance noise types
            train_dataset = MyCIFAR10(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=None,
                                    noisy_targets=noisy_targets)
            train_dataset.trans_matrix = np.eye(10)
            

        eval_dataset = MyCIFAR10(root=root,
                                 train=False,
                                 transform=eval_transform)
    
    return train_dataset, eval_dataset

def cifar100(root, noise_type, noise_rate, seed, tuning=False,saved_labels_file=None):
    # Initialize variables
    noisy_targets = None
    T = None
    
    # If saved labels file is provided, load it directly
    if saved_labels_file is not None and os.path.exists(saved_labels_file):
        # We'll load this later after we know it's not tuning mode
        pass
    elif noise_type == 'sym':
        T = get_sym_T(noise_rate, 100)
    elif 'asym' in noise_type:
        T = get_asym_T_cifar100(noise_rate, noise_type=='asym_single')
    elif noise_type == 'human':
        noisy_targets = torch.load(CIFAR100_HUMAN_NOISE_PATH,weights_only=False)['noisy_label']
    elif noise_type == 'instance':
        #use instance-dependent noise
        #load cifar100 training set without any noise
        base_dataset = datasets.CIFAR100(root=root, train=True, download=True)
        base_data = base_dataset.data
        base_labels = np.array(base_dataset.targets)
        #convert to torch tensor
        base_data = torch.from_numpy(base_data).float()
        base_labels = torch.from_numpy(base_labels)
        feature_size = 3*32*32
        noisy_targets = get_instance_noisy_label(noise_rate, base_dataset, base_labels, 100, feature_size, 0.1, seed)
    elif noise_type == 'block':
        T = np.eye(100)
        for i in range(20):
            for j in range(5):
                for k in range(5):
                    if j != k:
                        T[i*5 + j][i*5 + k] = noise_rate / 4.0
                T[i*5 + j][i*5 + j] = 1.0 - noise_rate
    else:
        raise ValueError('Wrong noise type! Must be either sym, asym, human, instance or block')
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    
    if tuning:
        train_dataset = MyCIFAR100(root=root,
                                   train=True,
                                   transform=train_transform,
                                   trans_matrix=T)
        num_train = int(len(train_dataset) * 0.9)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        train_dataset.trans_matrix = T

        eval_dataset = MyCIFAR100(root=root,
                                  train=True,
                                  transform=eval_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))

    else:
        # Load saved labels if provided
        if saved_labels_file is not None and os.path.exists(saved_labels_file):
            temp_dataset = datasets.CIFAR100(root=root, train=True, download=True)
            dataset_size = len(temp_dataset)
            noisy_targets = load_saved_labels(saved_labels_file, dataset_size)
            print(f"Loaded {len(noisy_targets)} labels from {saved_labels_file}")
            train_dataset = MyCIFAR100(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=None,
                                    noisy_targets=noisy_targets)
            # Set identity matrix for trans_matrix, needed for selalpha* methods
            train_dataset.trans_matrix = np.eye(100)
        elif noise_type not in ['human','instance']:
            train_dataset = MyCIFAR100(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=T)
        else:
            # For human/instance noise types
            train_dataset = MyCIFAR100(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=None,
                                    noisy_targets=noisy_targets)
            train_dataset.trans_matrix = np.eye(100)
        
        eval_dataset = MyCIFAR100(root=root,
                                  train=False,
                                  transform=eval_transform)

    return train_dataset, eval_dataset
