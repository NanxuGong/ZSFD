from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC,OneClassSVM
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.naive_bayes import GaussianNB
import numpy as np
import warnings
from util import *
from scipy.linalg import cholesky, svd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from feature_extract import *
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F
import seaborn as sns

from matplotlib import rcParams
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cpu')

warnings.filterwarnings("ignore")


def creat_dataset(test_index = [8, 12, 14]):
    path = './TE_mat_data/'
    print("loading data...")

    fault1 = loadmat(path + 'd01.mat')['data']
    fault2 = loadmat(path + 'd02.mat')['data']
    fault3 = loadmat(path + 'd03.mat')['data']
    fault4 = loadmat(path + 'd04.mat')['data']
    fault5 = loadmat(path + 'd05.mat')['data']
    fault6 = loadmat(path + 'd06.mat')['data']
    fault7 = loadmat(path + 'd07.mat')['data']
    fault8 = loadmat(path + 'd08.mat')['data']
    fault9 = loadmat(path + 'd09.mat')['data']
    fault10 = loadmat(path + 'd10.mat')['data']
    fault11 = loadmat(path + 'd11.mat')['data']
    fault12 = loadmat(path + 'd12.mat')['data']
    fault13 = loadmat(path + 'd13.mat')['data']
    fault14 = loadmat(path + 'd14.mat')['data']
    fault15 = loadmat(path + 'd15.mat')['data']

    attribute_matrix_ = pd.read_excel('./attribute_matrix.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values

    train_index = list(set(np.arange(15)) - set(test_index))

    test_index.sort()
    train_index.sort()

    print("test classes: {}".format(test_index))
    print("train classes: {}".format(train_index))

    data_list = [fault1, fault2, fault3, fault4, fault5,
                 fault6, fault7, fault8, fault9, fault10,
                 fault11, fault12, fault13, fault14, fault15]

    trainlabel = []
    train_attributelabel = []
    traindata = []
    for item in train_index:
        trainlabel += [item] * 480
        train_attributelabel += [attribute_matrix[item, :]] * 480
        traindata.append(data_list[item])
    trainlabel = np.row_stack(trainlabel)
    train_attributelabel = np.row_stack(train_attributelabel)
    traindata = np.column_stack(traindata).T

    testlabel = []
    test_attributelabel = []
    testdata = []
    for item in test_index:
        testlabel += [item] * 480
        test_attributelabel += [attribute_matrix[item, :]] * 480
        testdata.append(data_list[item])
    testlabel = np.row_stack(testlabel)
    test_attributelabel = np.row_stack(test_attributelabel)
    testdata = np.column_stack(testdata).T

    return traindata, trainlabel, train_attributelabel, \
           testdata, testlabel, test_attributelabel, \
           attribute_matrix_.iloc[test_index,:], attribute_matrix_.iloc[train_index, :],attribute_matrix_

def feature_extraction(traindata, testdata, train_attributelabel, test_attributelabel):
    trainfeatures = []
    testfeatures = []
    for i in range(train_attributelabel.shape[1]):
        spca = DSPCA(20)
        spca.fit(traindata, train_attributelabel[:, i])
        # print(traindata.shape)
        # print(train_attributelabel.shape)
        trainfeatures.append(spca.transform(traindata))
        testfeatures.append(spca.transform(testdata))
    return np.column_stack(trainfeatures), np.column_stack(testfeatures)

def pre_model(model, traindata, trainlabel, train_attributelabel, testdata, testlabel, test_attributelabel,
              attribute_matrix):
    model_dict = {'SVC_linear': SVC(kernel='linear', C=1), 'lr': LogisticRegression(), 'SVC_rbf': SVC(kernel='rbf'),
                  'rf': RandomForestClassifier(n_estimators=50), 'Ridge': Ridge(alpha=1), 'NB': GaussianNB(),
                  'Lasso': Lasso(alpha=0.1),'if':IsolationForest(contamination = 'auto'),'ocsvm':OneClassSVM(nu=0.4)}

    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            # print(testdata.shape)
            clf.fit(np.asarray(traindata), np.asarray(train_attributelabel[:, i]))
            res = clf.predict(np.asarray(testdata))
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(attribute_matrix.values - pre_res), axis=1)).argmin()
        label_lis.append(attribute_matrix.index[loc] - 1)
    label_lis = np.mat(np.row_stack(label_lis))
    # print(model)
    # accuracy(np.asarray(label_lis), np.asarray(testlabel))
    return label_lis, testlabel


def pre_model_proba(model, traindata, trainlabel, train_attributelabel, testdata, testlabel, test_attributelabel,
                    attribute_matrix):
    model_dict = {'SVC_linear': SVC(kernel='linear', probability=True), 'lr': LogisticRegression(),
                  'SVC_rbf': SVC(kernel='rbf', probability=True),
                  'rf': RandomForestClassifier(n_estimators=10), 'Ridge': Ridge(alpha=1), 'NB': GaussianNB(),
                  'Lasso': Lasso(alpha=0.1)}

    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata)
            res = clf.predict(testdata)
            res = np.where(res == -1, 1, 0)
            # res = clf.predict_proba(testdata)
        else:
            res = np.ones((testdata.shape[0], 2))
            res[:, 1] = res[:, 1] * 0.001
            res[:, 0] = res[:, 0] * 0.999
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        res_list = [1] * attribute_matrix.shape[0]
        pre_res = np.ravel(test_pre_attribute[i, :])
        for j in range(attribute_matrix.shape[0]):
            for k in range(attribute_matrix.shape[1]):
                if attribute_matrix.iloc[j, k] == 0:
                    res_list[j] = res_list[j] * pre_res[k * 2]
                else:
                    res_list[j] = res_list[j] * pre_res[k * 2 + 1]
        loc = np.array(res_list).argmax()
        label_lis.append(attribute_matrix.index[loc] - 1)
    label_lis = np.mat(np.row_stack(label_lis))
    print(model)
    accuracy(label_lis, testlabel)
    return label_lis, testlabel

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
            
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(0.1),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,input_dim),
            nn.ReLU()
        )


    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, z):
        return self.decoder(z)
    
    def compability_loss(self,z1,z2):
        N,D=z1.shape

        c=self.bn2(z1).T @ self.bn2(z2)/N

        on_diag=torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag=off_diagonal(c).pow_(2).sum()
        loss=on_diag+0.25*off_diag

        return loss

    def forward(self, x):
        z = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, z

class Classify(nn.Module):
    def __init__(self,latent_dim):
        self.latent_dim = latent_dim
        super(Classify,self).__init__()
        self.rl = nn.ReLU()
        self.fc1 = nn.Linear(latent_dim,latent_dim)
        self.fc2 = nn.Linear(latent_dim,20)
        self.bn = nn.BatchNorm1d(latent_dim)
        self.bn2 = nn.BatchNorm1d(20)
        self.sigmoid = nn.Sigmoid()
        self.do = nn.Dropout(0.2)
    def forward(self,x):
        x = self.fc1(x)
        x = self.do(x)
        x = self.bn(x)
        x = self.rl(x)
        x = self.fc2(x)
        x = self.bn2(x)
        # x = self.sigmoid(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data,label,glabel):
        # scaler = StandardScaler()
        # self.data = scaler.fit_transform(data)
        self.data = torch.tensor(data).type(torch.FloatTensor).to(device)
        self.label = label
        self.label = torch.tensor(self.label).type(torch.FloatTensor).to(device)
        self.glabel = torch.tensor(glabel).type(torch.FloatTensor).to(device)
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        z = self.glabel[index]
        return x,y,z

    def __len__(self):
        return len(self.data)

def cosine_similarity(A, B):
    A = F.normalize(A, dim=1)  
    B = F.normalize(B, dim=1)  
    cos_similarity = torch.nn.functional.cosine_similarity(A, B)
    return cos_similarity

class CenterLoss(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
 
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
 
    def forward(self, x, labels):

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
 
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes
        labels = labels.expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
 
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
 
        return loss
def remap_labels(labels):
    labels = np.squeeze(labels)
    unique_labels = np.unique(labels)  
    old_to_new_mapping = {label: i for i, label in enumerate(unique_labels)}  

    new_labels = np.array([old_to_new_mapping[label] for label in labels]) 

    return new_labels

def run():

    data_list = [0,5,13]
    data_list = [3,6,9]
    # data_list = [7,10,11]
    # data_list = [1,2,4]
    # data_list = [8,12,14]
    batch_size = 64
    device = torch.device("cpu")
    hidden_dim = 256
    latent_dim = 100
    traindata, trainlabel, train_attributelabel, testdata, testlabel, \
            test_attributelabel, attribute_matrix, train_attribute_matrix,all_attribute_matrix = creat_dataset(data_list)
    traindata, testdata = feature_extraction(traindata,testdata,train_attributelabel,test_attributelabel)
    trainlabel_ = remap_labels(trainlabel)

    dataset = CustomDataset(traindata,train_attributelabel,trainlabel_)
    num_epochs = 300
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = AE(input_dim=400, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model = model.to(device)
    classifier = Classify(latent_dim = latent_dim).to(device)

    # criterion = nn.MSELoss()
    centerloss = CenterLoss(num_classes=12,feat_dim=100)
    # criterion_bce = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    optimizer = optim.RMSprop(list(model.parameters())+list(classifier.parameters()), lr=1e-2)

    accs = []
    max_acc = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        acc = 0
        model.train()
        for inputs,labels,glabel in dataloader:
            reconstruction, z = model(inputs)
            output = classifier(z)
            reconstruction_loss = F.cosine_similarity(inputs, reconstruction)
            reconstruction_loss = 1 - reconstruction_loss.mean()
            p_compatibility = torch.sum(output * labels, 1, keepdim=True)
            n_compatibility = output @ torch.tensor(train_attribute_matrix.values.T).type(torch.FloatTensor)
            # print(p_compatibility.shape)
            loss_nce = - p_compatibility + torch.log(torch.sum(n_compatibility.exp(), 1, keepdim=True))
            loss_nce = loss_nce.mean()
            center_loss = centerloss(z,glabel.unsqueeze(1))
            loss = 0.05*reconstruction_loss+0.5*center_loss+0.5*loss_nce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        model.eval()
        with torch.no_grad():
            # print(111)
            # scaler = StandardScaler()


            train_recon,traindata_ = model(torch.tensor(traindata).type(torch.FloatTensor).to(device))
        
            test_recon,testdata_= model(torch.tensor(testdata).type(torch.FloatTensor).to(device))
            traindata_tensor = torch.tensor(traindata)
            testdata_tensor = torch.tensor(testdata)
            train_fea = classifier(traindata_)
            test_fea = classifier(testdata_)
            traindata_ = torch.cat([traindata_,traindata_tensor,train_recon,train_fea],dim=1).detach().cpu().numpy()
            testdata_ = torch.cat([testdata_,testdata_tensor,test_recon,test_fea],dim=1).detach().cpu().numpy()
            # traindata_ = traindata_.detach().cpu().numpy()
            # testdata_ = testdata_.detach().cpu().numpy()
            # print(traindata_)
            # df = pd.DataFrame(traindata)
            # df.to_excel("output.xlsx",index=False)
            label_lis, testlabel = pre_model('NB', traindata_, trainlabel, train_attributelabel, testdata_, testlabel,
                    test_attributelabel, attribute_matrix)
            # print(label_lis.shape)
            # print(testlabel.shape)
            acc = accuracy_score(np.asarray(label_lis), np.asarray(testlabel))
            accs.append(acc)

            if acc>max_acc:
                max_acc = acc
                torch.save(model.state_dict(), 'model.pth')
                torch.save(classifier.state_dict(),'classifier.pth')
        print('epoch:{:d}, rec_loss:{:.6f}, nce_loss:{:.6f}, center_loss{:.6f}, now_acc:{:.4f}, best_acc:{:.4f}'.format(epoch,reconstruction_loss,loss_nce,center_loss,acc,max_acc))
    # plt.plot(accs)
    # plt.title('my')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.show()

if __name__ == '__main__':
    run()




