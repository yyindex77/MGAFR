import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

import config
from dataloader_cmumosi import CMUMOSIDataset
from model import MGAFR
from utils import MaskedReconLoss
from utils import get_contrastive_loss
from utils import random_mask
from utils import eval_model


def get_loaders(audio_root, text_root, video_root, dataset, batch_size, num_workers):
    
    dataset = CMUMOSIDataset(label_path=config.PATH_TO_LABEL[dataset],
                                audio_root=audio_root,
                                text_root=text_root,
                                video_root=video_root)
    
    all_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=dataset.collate_fn,
                                num_workers=num_workers,
                                pin_memory=False,
                                shuffle=True)
    
    adim, tdim, vdim = dataset.get_featDim()
    
    return adim, tdim, vdim, all_loader

## gain input features: ?*[seqlen, batch, dim]
def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    
    input_features = [] 
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2) # [seqlen, batch, featdim=adim+tdim+vdim]
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    
    tmask = qmask.transpose(0, 1) # [batch, seqlen] -> [seqlen, batch] 
    tmask = tmask.unsqueeze(2).repeat(1,1,featdim) # -> [seqlen, batch, featdim]
    select_feat = torch.where(tmask==0, feat1, feat2) # -> [seqlen, batch, featdim]
    input_features.append(select_feat) # 1 * [seqlen, batch, dim]
    return input_features

def train_MGAFR(args, model, rec_loss, dataloader, mask_rate=None, optimizer=None, train=False):
    
    labels, masks = [], []
    savelabels, savehiddens = [], []
    losses, losses1, losses2, losses3 = [], [], [], []

    save_raw_audio,save_raw_video,save_raw_text = [], [], []
    save_recon_audio,save_recon_video,save_recon_text = [], [], []
    
    dataset = args.dataset
    cuda = torch.cuda.is_available()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
        
    for data in dataloader:
        
        if train: optimizer.zero_grad()
        
        ## read dataloader
        """
        audio_host, text_host, visual_host: [seqlen, batch, dim]
        audio_guest, text_guest, visual_guest: [seqlen, batch, dim]
        qmask: speakers, [batch, seqlen]
        umask: has utt, [batch, seqlen]
        label: [batch, seqlen]
        """

        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        
        audio_host, text_host, visual_host = audio_host.cuda(), text_host.cuda(), visual_host.cuda()
        audio_guest, text_guest, visual_guest = audio_guest.cuda(), text_guest.cuda(), visual_guest.cuda()
        qmask, umask, label = qmask.cuda(), umask.cuda(), label.cuda()
        
        adim, tdim, vdim = audio_host.size(2), text_host.size(2), visual_host.size(2)
        seqlen, batch = audio_host.size(0), audio_host.size(1)
        view_num = 3
        
        ## using cmp-net masking manner [at least one view exists]
        """
        ?_?_mask: [seqlen, batch, dim]   => gain mask
        masked_?_?: [seqlen, batch, dim] => masked features

        # if audio_feature is None: audio_feature = text_feature
        # if text_feature is None: text_feature = audio_feature
        # if video_feature is None: video_feature = text_feature
        # mask sure, same mask for same features [include padded features]
        """
        
        matrix = random_mask(view_num, seqlen*batch, mask_rate) # [seqlen*batch, view_num]
        audio_host_mask = np.reshape(matrix[:, 0], (seqlen, batch, 1)) 
        text_host_mask = np.reshape(matrix[:, 1], (seqlen, batch, 1))
        visual_host_mask = np.reshape(matrix[:, 2], (seqlen, batch, 1))
        audio_host_mask = torch.LongTensor(audio_host_mask).cuda()
        text_host_mask = torch.LongTensor(text_host_mask).cuda()
        visual_host_mask = torch.LongTensor(visual_host_mask).cuda()
        
        
        # guest mask
        view_num = 3
        matrix = random_mask(view_num, seqlen*batch, mask_rate) # [seqlen*batch, view_num]
        audio_guest_mask = np.reshape(matrix[:, 0], (seqlen, batch, 1)) 
        text_guest_mask = np.reshape(matrix[:, 1], (seqlen, batch, 1))
        visual_guest_mask = np.reshape(matrix[:, 2], (seqlen, batch, 1))
        audio_guest_mask = torch.LongTensor(audio_guest_mask).cuda()
        text_guest_mask = torch.LongTensor(text_guest_mask).cuda()
        visual_guest_mask = torch.LongTensor(visual_guest_mask).cuda()
        if view_num == 2: assert mask_rate <= 0.500001, f'Warning: at least one view exists'
        if view_num == 3: assert mask_rate <= 0.700001, f'Warning: at least one view exists'
        
        
        masked_audio_host = audio_host * audio_host_mask 
        masked_audio_guest = audio_guest * audio_guest_mask 
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask
        

        input_features = generate_inputs(audio_host, text_host, visual_host, \
                                         audio_guest, text_guest, visual_guest, qmask)
        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host, \
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask, \
                                                audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)
        

        '''
        # input_features, masked_input_features, input_features_mask: ?*[seqlen, batch, dim]
        # qmask: speakers, [batch, seqlen]
        # umask: has utt, [batch, seqlen]
        # label: [batch, seqlen]
        # input_features_recon # padded, ?*[seqlen, batch, dim]
        '''
        
        recon_input_features, hidden, hidden_other = model(masked_input_features, umask, input_features_mask)
        
        ## gain saved results [utterance-level]
        tempseqlen = np.sum(umask.cpu().data.numpy(), 1) # [batch]
        temphidden = hidden.transpose(0,1).cpu().data.numpy() # [batch, seqlen, featdim]
        templabel = label.cpu().data.numpy() # [batch, seqlen]

        temp_raw_audio = audio_host.transpose(0,1).cpu().data.numpy()
        temp_raw_text = text_host.transpose(0,1).cpu().data.numpy()
        temp_raw_video = visual_host.transpose(0,1).cpu().data.numpy()
        temp_recon_audio = recon_input_features[0][:,:,0:adim].transpose(0,1).cpu().data.numpy()
        temp_recon_text = recon_input_features[0][:,:,adim:adim+tdim].transpose(0,1).cpu().data.numpy()
        temp_recon_video = recon_input_features[0][:,:,adim+tdim:].transpose(0,1).cpu().data.numpy()

        
        for ii in range(len(tempseqlen)): # utt_number for each conversation
            xii = int(tempseqlen[ii])
            itemhidden = temphidden[ii][:xii, :] # [seqlen, featdim]
            itemlabel  = templabel[ii][:xii]     # [len, ]
            savehiddens.append(itemhidden)
            savelabels.append(itemlabel)
            
            item_raw_audio = temp_raw_audio[ii][:int(tempseqlen[ii]), :]
            item_raw_text = temp_raw_text[ii][:int(tempseqlen[ii]), :]
            item_raw_video = temp_raw_video[ii][:int(tempseqlen[ii]), :]
            save_raw_audio.append(item_raw_audio)
            save_raw_text.append(item_raw_text)
            save_raw_video.append(item_raw_video)
            item_recon_audio = temp_recon_audio[ii][:int(tempseqlen[ii]), :]
            item_recon_text = temp_recon_text[ii][:int(tempseqlen[ii]), :]
            item_recon_video = temp_recon_video[ii][:int(tempseqlen[ii]), :]
            save_recon_audio.append(item_recon_audio)
            save_recon_text.append(item_recon_text)
            save_recon_video.append(item_recon_video)
            
        ## calculate loss 
        contrastive_loss = get_contrastive_loss(hidden_other, umask)
        reconstruct_loss = rec_loss(recon_input_features, input_features, input_features_mask, umask, adim, tdim, vdim) * args.recon_weight
        featureInfo_loss = hidden_other[3] * args.inf_weight
        
        loss1 = contrastive_loss
        loss2 = reconstruct_loss
        loss3 = featureInfo_loss
        
        loss = loss1 + loss2 + loss3
        
        ## save batch results
        labels.append(label.view(-1).data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())
        losses1.append(loss1.item()*masks[-1].sum())
        losses2.append(loss2.item()*masks[-1].sum())
        losses3.append(loss3.item()*masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()
    
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_loss1 = round(np.sum(losses1)/np.sum(masks), 4)
    avg_loss2 = round(np.sum(losses2)/np.sum(masks), 4)
    avg_loss3 = round(np.sum(losses3)/np.sum(masks), 4)
        
    
    save_dict = {}
    save_dict["save_raw_audio"] = np.concatenate(save_raw_audio,axis=0)
    save_dict["save_raw_text"] = np.concatenate(save_raw_text,axis=0)
    save_dict["save_raw_video"] = np.concatenate(save_raw_video,axis=0)
    save_dict["save_recon_audio"] = np.concatenate(save_recon_audio,axis=0)
    save_dict["save_recon_text"] = np.concatenate(save_recon_text,axis=0)
    save_dict["save_recon_video"] = np.concatenate(save_recon_video,axis=0)
    save_dict["savehiddens"] = np.concatenate(savehiddens,axis=0)
    save_dict["savelabels"] = np.concatenate(savelabels,axis=0)

    return [avg_loss, avg_loss1, avg_loss2, avg_loss3], save_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT', help='audio feature name')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT', help='text feature name')
    parser.add_argument('--video-feature', type=str, default='manet_UTT', help='video feature name')
    parser.add_argument('--dataset', type=str, default='CMUMOSI', help='dataset type')

    ## Params for training
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs')
    parser.add_argument('--mask-type', type=str, default='constant-0.0', help='mask rate [0~1] for input argumentation: constant-float; linear; convex; concave')
    
    parser.add_argument('--recon_weight', type=float, default=1000, metavar='LR', help='recon loss weight')
    parser.add_argument('--inf_weight', type=float, default=10, metavar='LR', help='informative loss weight')
    
    args = parser.parse_args()

    args.n_classes = 1
    args.n_speakers = 1
    
    cuda = torch.cuda.is_available()


    print (f'====== Reading Data =======')
    audio_feature = args.audio_feature
    text_feature = args.text_feature
    video_feature = args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), f'features not exist!'
    adim, tdim, vdim, all_loader = get_loaders( audio_root = audio_root,
                                                text_root  = text_root,
                                                video_root = video_root,
                                                batch_size = args.batch_size,
                                                dataset = args.dataset,
                                                num_workers = 0)

    
    print (f'====== Training and Evaluation =======')
    
    print (f'Step1: build model')
    model = MGAFR(adim, tdim, vdim,args.n_classes)
    rec_loss = MaskedReconLoss()

    if cuda:
        model.cuda()
        rec_loss.cuda()
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    mask_rate = float(args.mask_type.split('-')[-1])
    
    print (f'Step2: training')
    best_model = copy.deepcopy(model)
    
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        assert args.mask_type.startswith('constant'), f'mask_type should be constant-x.x'
        
        train_loss, trainsave = train_MGAFR(args, model, rec_loss, all_loader, mask_rate=mask_rate, optimizer=optimizer, train=True)
        
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print(f'epoch:{epoch+1}; train_loss:{train_loss[0]}; train_loss1:{train_loss[1]}; train_loss2:{train_loss[2]}; train_loss3:{train_loss[3]}')
        
        if best_loss > train_loss[0]:
            best_loss = train_loss[0]
            best_trainsave = trainsave
        
    test_results, test_save_result = eval_model(best_trainsave, args.dataset)
    

    print (f'====== Saving =======')
    
    print(test_results)
    log_result_file = f'./logs/{args.dataset.lower()}_result.txt'
    with open(log_result_file, 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'\n')
        f.write(f'./logs/{args.dataset.lower()}_mask:{mask_rate}_result.npz'+'\n')
        f.write(str(test_results)+'\n')
        f.write('\n')
    
    np.savez(f'./logs/{args.dataset.lower()}_mask:{mask_rate}_result.npz', **test_save_result)

    print(f'====== Finish =======')
    