import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


## follow cpm-net's masking manner
def random_mask(view_num, input_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    """

    assert missing_rate is not None
    one_rate = 1 - missing_rate

    if one_rate <= (1 / view_num): 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(input_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(input_len, view_num)) # [samplenum, viewnum] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    if input_len < 32:
        alldata_len = 32
    else:
        alldata_len = input_len
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    
    matrix = matrix[:input_len, :]
    return matrix

def get_contrastive_loss(hidden_other,umask):
    a = hidden_other[0]
    t = hidden_other[1]
    v = hidden_other[2]
    
    a = a.contiguous().view(-1,a.shape[2])
    t = t.contiguous().view(-1,t.shape[2])
    v = v.contiguous().view(-1,v.shape[2])
    
    umask = umask.view(-1)
    nonzero_indices = torch.nonzero(umask).view(-1)
    
    a = a[nonzero_indices]
    t = t[nonzero_indices]
    v = v[nonzero_indices]

    loss1 = get_contrastive_loss_one2one(a,t)
    loss2 = get_contrastive_loss_one2one(a,v)
    loss3 = get_contrastive_loss_one2one(t,v)
    return loss1 + loss2 + loss3
    
def get_contrastive_loss_one2one(z_i, z_j):
    batch_size = z_i.shape[0]
    temperature = 0.5
    mask = mask_correlated_samples(batch_size)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    
    N = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)

    sim = torch.matmul(z, z.T) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_samples = sim[mask].reshape(N, -1)
    
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = criterion(logits, labels)
    loss /= N

    return loss

def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    mask = mask.bool()
    return mask

class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_regression,
                'MOSEI': self.__eval_mosei_regression,
                'SIMS': self.__eval_sims_regression
            }
        else:
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_classification,
                'MOSEI': self.__eval_mosei_classification,
                'SIMS': self.__eval_sims_classification
            }

    def __eval_mosi_classification(self, y_pred, y_true):
        """
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2   
        }
        """
        # three classes
        y_pred_3 = np.argmax(y_pred, axis=1)
        Mult_acc_3 = accuracy_score(y_pred_3, y_true)
        F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')
        # two classes 
        y_pred = np.array([[v[0], v[2]] for v in y_pred])
        # with 0 (<= 0 or > 0)
        y_pred_2 = np.argmax(y_pred, axis=1)
        y_true_2 = []
        for v in y_true:
            y_true_2.append(0 if v <= 1 else 1)
        y_true_2 = np.array(y_true_2)
        Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
        # without 0 (< 0 or > 0)
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
        y_pred_2 = y_pred[non_zeros]
        y_pred_2 = np.argmax(y_pred_2, axis=1)
        y_true_2 = y_true[non_zeros]
        Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

        eval_results = {
            "Has0_acc_2":  round(Has0_acc_2, 4),
            "Has0_F1_score": round(Has0_F1_score, 4),
            "Non0_acc_2":  round(Non0_acc_2, 4),
            "Non0_F1_score": round(Non0_F1_score, 4),
            "Acc_3": round(Mult_acc_3, 4),
            "F1_score_3": round(F1_score_3, 4)
        }
        return eval_results
    
    def __eval_mosei_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __eval_sims_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.reshape(-1)
        test_truth = y_true.reshape(-1)

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)


        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')
        
        eval_results = {
            "Has0_acc_2":  round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2":  round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results


    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.reshape(-1)
        test_truth = y_true.reshape(-1)
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
        
        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
 
        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": mult_a2,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "F1_score": f_score,
            "MAE": mae,
            "Corr": corr, # Correlation Coefficient
        }
        return eval_results
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]

class Linear_Network(nn.Module):
    def __init__(self, input_size, n_classes):
        super(Linear_Network, self).__init__()
        
        self.f = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, xs, ys):
        cls_result = self.f(xs)
        cls_result = cls_result.squeeze(dim=1)
        loss = self.loss(cls_result, ys)
        return cls_result,loss
        
def eval_model(trainsave, data_name):

    if data_name == "CMUMOSI":
        train_num = 1284
        val_num = 229
        test_num = 686
        epoch = 1000
    elif data_name == "CMUMOSEI":
        train_num = 16326
        val_num = 1871
        test_num = 4659
        epoch = 500

    real_save_train,real_save_val,real_save_test = {},{},{}
    
    all_hidden = trainsave["savehiddens"]
    real_save_train["savehiddens"] = all_hidden[:train_num]
    real_save_val["savehiddens"] = all_hidden[train_num:train_num+val_num]
    real_save_test["savehiddens"] = all_hidden[train_num+val_num:]

    all_label = trainsave["savelabels"]
    real_save_train["savelabels"] = all_label[:train_num]
    real_save_val["savelabels"] = all_label[train_num:train_num+val_num]
    real_save_test["savelabels"] = all_label[train_num+val_num:]


    all_hidden = torch.tensor(all_hidden).cuda()
    all_label = torch.tensor(all_label).cuda()

    H_train = all_hidden[:train_num]
    train_labels = all_label[:train_num]
    H_val = all_hidden[train_num:train_num+val_num]
    val_labels = all_label[train_num:train_num+val_num]
    H_test = all_hidden[train_num+val_num:]
    test_labels = all_label[train_num+val_num:]

    linear_batch = 2048
    train_dataset = TensorDataset(H_train,train_labels)
    linear_train_loader = DataLoader(dataset=train_dataset, batch_size=linear_batch, shuffle=False)
    val_dataset = TensorDataset(H_val,val_labels)
    linear_val_loader = DataLoader(dataset=val_dataset, batch_size=linear_batch, shuffle=False)
    test_dataset = TensorDataset(H_test,test_labels)
    linear_test_loader = DataLoader(dataset=test_dataset, batch_size=linear_batch, shuffle=False)
    
    linear_model = Linear_Network(H_train.shape[1],1).cuda()
    linear_optimizer = optim.Adam(linear_model.parameters(), lr=0.0001, weight_decay=0.00001)
    
    old_metrics = {}
    old_metrics['Has0_acc_2'] = 0

    for epoch in range(epoch):
        train_results,train_preds = train_or_eval_linear(linear_model, linear_train_loader, optimizer=linear_optimizer, train=True)
        val_results,val_preds = train_or_eval_linear(linear_model, linear_val_loader, optimizer=None, train=False)
        test_results,test_preds = train_or_eval_linear(linear_model, linear_test_loader, optimizer=None, train=False)
        
        if old_metrics['Has0_acc_2'] < val_results['Has0_acc_2']:
            old_metrics = test_results
            
    return old_metrics, {"train":real_save_train, "val":real_save_val, "test":real_save_test}

def train_or_eval_linear(model, dataloader, optimizer=None, train=True):
    cuda = torch.cuda.is_available()
    preds, labels = [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for xs, ys in dataloader:
        if train: optimizer.zero_grad()
        
        xs = xs.to(torch.float32)
        ys = ys.to(torch.float32)
        
        pred, loss = model(xs, ys)

        preds.append(pred.data.cpu().numpy())
        labels.append(ys.data.cpu().numpy())
        
        if train:
            loss.backward()
            optimizer.step()
        
    preds = np.concatenate(preds,axis=0)
    labels = np.concatenate(labels,axis=0)
    
    metrics = MetricsTop("regression").getMetics("mosi")
    eval_results = metrics(preds, labels)
        
    return eval_results,preds


class MaskedReconLoss(nn.Module):

    def __init__(self):
        super(MaskedReconLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, recon_input, target_input, input_mask, umask, adim, tdim, vdim):
        """ ? => refer to spk and modality
        recon_input  -> ? * [seqlen, batch, dim]
        target_input -> ? * [seqlen, batch, dim]
        input_mask   -> ? * [seqlen, batch, dim]
        umask        -> [batch, seqlen]
        """
        assert len(recon_input) == 1
        recon = recon_input[0] # [seqlen, batch, dim]
        target = target_input[0] # [seqlen, batch, dim]
        mask = input_mask[0] # [seqlen, batch, 3]
        
        recon  = torch.reshape(recon, (-1, recon.size(2)))   # [seqlen*batch, dim]
        target = torch.reshape(target, (-1, target.size(2))) # [seqlen*batch, dim]
        mask   = torch.reshape(mask, (-1, mask.size(2)))     # [seqlen*batch, 3] 1(exist); 0(mask)
        umask = torch.reshape(umask.permute(1,0), (-1, 1)) # [seqlen*batch, 1]

        A_rec = recon[:, :adim]
        L_rec = recon[:, adim:adim+tdim]
        V_rec = recon[:, adim+tdim:]
        A_full = target[:, :adim]
        L_full = target[:, adim:adim+tdim]
        V_full = target[:, adim+tdim:]

        A_miss_index = torch.reshape(mask[:, 0], (-1, 1))
        L_miss_index = torch.reshape(mask[:, 1], (-1, 1))
        V_miss_index = torch.reshape(mask[:, 2], (-1, 1))
        
        loss_recon1 = self.loss(A_rec*umask, A_full*umask) * A_miss_index
        loss_recon2 = self.loss(L_rec*umask, L_full*umask) * L_miss_index
        loss_recon3 = self.loss(V_rec*umask, V_full*umask) * V_miss_index

        loss_recon1 = torch.sum(loss_recon1) / adim
        loss_recon2 = torch.sum(loss_recon2) / tdim
        loss_recon3 = torch.sum(loss_recon3) / vdim
        loss_recon = (loss_recon1 + loss_recon2 + loss_recon3) / torch.sum(umask)

        return loss_recon
    