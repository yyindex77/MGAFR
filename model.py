import torch
import torch.nn as nn

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x
    
class MGAFR(nn.Module):
    def __init__(self,adim, tdim, vdim,n_classes):
        super(MGAFR, self).__init__()
        
        self.adim = adim
        self.tdim = tdim
        self.vdim = vdim
        
        self.encode_dim = 2048
        
        self.Wa = nn.Sequential(
            nn.Linear(adim, self.encode_dim),
        )
        self.Wt = nn.Sequential(
            nn.Linear(tdim, self.encode_dim),
        )
        self.Wv = nn.Sequential(
            nn.Linear(vdim, self.encode_dim),
        )

        self.decoder_audio = nn.Sequential(
            nn.Linear(self.encode_dim, adim),
        )
        self.decoder_text = nn.Sequential(
            nn.Linear(self.encode_dim, tdim),
        )
        self.decoder_video = nn.Sequential(
            nn.Linear(self.encode_dim, vdim),
        )
        
        self.normalization = "NormAdj"
        self.degree = 1
        self.alpha = 0.75 
        self.k = 4
        
        self.k1 = 1
        self.k2 = 1
        self.mu = 0.5 
        
        self.weight_a = LinearLayer(self.encode_dim, self.encode_dim)
        self.weight_t = LinearLayer(self.encode_dim, self.encode_dim)
        self.weight_v = LinearLayer(self.encode_dim, self.encode_dim)
        
    def aug_normalized_adjacency(self,adj):
        adj = adj + torch.eye(adj.shape[0]).cuda()
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.where(row_sum != 0, 1.0 / torch.sqrt(row_sum), torch.zeros(1).cuda())
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt).cuda()
        adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return adj

    def normalized_adjacency(self,adj):
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.where(row_sum != 0, 1.0 / torch.sqrt(row_sum), torch.zeros(1).cuda())
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt).cuda()
        adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return adj

    def row_normalize(self,mx):
        rowsum = mx.sum(dim=1)
        r_inv = torch.where(rowsum != 0, 1.0 / rowsum, torch.zeros(1).cuda())
        r_mat_inv = torch.diagflat(r_inv).cuda()
        mx = r_mat_inv @ mx
        return mx

    def preprocess_citation(self,adj, features, normalization="FirstOrderGCN"):
        features = self.row_normalize(features) 
        if normalization=="AugNormAdj":
            adj = self.aug_normalized_adjacency(adj)
        elif normalization=="NormAdj":
            adj = self.normalized_adjacency(adj)
        else:
            print("Invalid normalization technique.")
        return adj, features

    def get_affinity_matrix(self,input_data,dim,mask_list):
        data = input_data.clone().detach()
        index = IndexFlatL2(data)
        k = self.k
        adj = torch.zeros((input_data.shape[0], input_data.shape[0])).cuda()
        
        for i in range(data.shape[0]):
            query_vector = data[i]
            adj[i][i] = 1.0
            if mask_list[i]==0:
                continue
            nearest_indices, nearest_distances = index.search(query_vector, k)

            for j in range(nearest_indices.shape[0]):
                neighbor_index = nearest_indices[j]
                if mask_list[neighbor_index]==0:
                    continue
                similarity = 1 / (1 + nearest_distances[j])  # 距离转换为相似度
                adj[i][neighbor_index] = similarity
                adj[neighbor_index][i] = similarity
        
        return adj,input_data
    
    
    def SelfFilter(self,features, adj, degree, alpha):
        adj, features = self.preprocess_citation(adj, features, self.normalization)
        features = torch.tensor(features, dtype=torch.float32) 
        emb = alpha * features
        for i in range(degree):
            features = torch.spmm(adj, features)
            emb = emb + (1-alpha)*features/degree
        return emb

    def delete_umask(self,a,umask):
        another_a = a[0][:int(sum(umask[0])),:]
        for batch_i in range(1,umask.shape[0]):
            seqlen_i = int(sum(umask[batch_i]))
            another_a = torch.cat([another_a,a[batch_i][:seqlen_i,:]],dim=0)
        return another_a

    def add_umask(self,a,umask):
        a_dim = a.shape[1]
        seqlen = umask.shape[1] #[batch, seqlen]
        seqlen_sum = int(sum(umask[0]))
        another_a = torch.cat([a[:seqlen_sum,:],torch.zeros(seqlen-seqlen_sum,a_dim).cuda()])
        another_a = torch.unsqueeze(another_a, dim=0)
        
        for batch_i in range(1,umask.shape[0]):
            seqlen_i = int(sum(umask[batch_i]))
            another_a_i = torch.cat([a[seqlen_sum:seqlen_sum+seqlen_i,:],torch.zeros(seqlen-seqlen_i,a_dim).cuda()])
            another_a_i = torch.unsqueeze(another_a_i, dim=0)
            another_a = torch.cat([another_a,another_a_i],dim=0)
            seqlen_sum = seqlen_sum+seqlen_i
            
        return another_a
    
    def MixedFilter(self, X, S, S_bar1,S_bar2, k1, k2, mu):
        H_low1 = self.LowPassFilter(X, S_bar1, k2)
        H_low2 = self.LowPassFilter(X, S_bar2, k1)
        H =  (1 - mu) * H_low1 + mu * H_low2
        return H
    
    def LowPassFilter(self, X, S, k1, p=0.5):
        I = torch.eye(S.shape[0]).cuda() 
        S = S + I
        S = self.normalize_matrix(S)
        L_S = I - S 

        H_low = X.clone() 
        for i in range(k1):
            H_low = (I - p * L_S).matmul(H_low)
            
        return H_low

    def normalize_matrix(self, A, eps=1e-12):
        D = torch.sum(A, dim=1) + eps
        D = torch.pow(D, -0.5)
        D[D == float('inf')] = 0
        D[D != D] = 0
        D = torch.diagflat(D).cuda()
        A = D @ A @ D
        return A

    def knn_fill(self,a,t,v,a_adj,t_adj,v_adj,features_mask_del_umask):
        new_features_list = [a,t,v]#[3,len,dim]
        new_adj_list = [a_adj,t_adj,v_adj]#[3,len,len]
        
        features_list = [a,t,v]#[3,len,dim]
        adj_list = [a_adj,t_adj,v_adj]#[3,len,len]
        #features_mask_del_umask：[len,3]
        n = a.shape[0]
        
        features_len = features_mask_del_umask.shape[0] #[len,3]
        for i in range(features_len): 
            for j in range(3): 
                if features_mask_del_umask[i][j]==0: 
                    non_mask_model = [] 
                    if features_mask_del_umask[i][(j+1)%3]!=0: 
                        non_mask_model.append((j+1)%3)
                    if features_mask_del_umask[i][(j+2)%3]!=0: 
                        non_mask_model.append((j+2)%3)
                    
                        
                    num_link = 0
                    for no_mask_model_index in non_mask_model: 
                        neighbor_indices = torch.nonzero(adj_list[no_mask_model_index][i]) 
                        neighbor_indices = neighbor_indices.permute(1,0) 
                        
                        for neighbor_indice in neighbor_indices[0]: 
                            if features_mask_del_umask[neighbor_indice][j] != 0:
                                new_features_list[j][i] = new_features_list[j][i] + features_list[j][neighbor_indice]
                                num_link+=1
                    
                        
                    if num_link!=0:
                        new_features_list[j][i] = new_features_list[j][i]/num_link
                    
                        for no_mask_model_index in non_mask_model: 
                            neighbor_indices = torch.nonzero(adj_list[no_mask_model_index][i])
                            neighbor_indices = neighbor_indices.permute(1,0)
                            
                            for neighbor_indice in neighbor_indices[0]:
                                if features_mask_del_umask[neighbor_indice][j] != 0:
                                    distance = torch.norm(new_features_list[j][i] - features_list[j][neighbor_indice], p=2) 
                                    
                                    new_adj_list[j][i][neighbor_indice] = 1 / (1 + distance)
                                    new_adj_list[j][neighbor_indice][i] = 1 / (1 + distance)
                                else:
                                    new_adj_list[j][i][neighbor_indice] = 1/n
                                    new_adj_list[j][neighbor_indice][i] = 1/n
                                    
                    else:
                        for no_mask_model_index in non_mask_model: 
                            neighbor_indices = torch.nonzero(adj_list[no_mask_model_index][i]).permute(1,0) 
                            
                            for neighbor_indice in neighbor_indices[0]:
                                new_adj_list[j][i][neighbor_indice] = 1/n
                                new_adj_list[j][neighbor_indice][i] = 1/n
                    
        return new_features_list[0],new_features_list[1],new_features_list[2],new_adj_list[0],new_adj_list[1],new_adj_list[2]
    
    def forward(self, inputfeats, umask, input_features_mask):
        inputfeats_tensor = inputfeats[0]
        a = inputfeats_tensor[:,:,0:self.adim].permute(1,0,2)
        t = inputfeats_tensor[:,:,self.adim:self.adim+self.tdim].permute(1,0,2)
        v = inputfeats_tensor[:,:,self.adim+self.tdim:].permute(1,0,2)
        raw_shape = a.shape
        #a:torch.Size([batch, seqlen, 512])，mask:torch.Size([batch, seqlen])
        
        features_mask = input_features_mask[0].permute(1,0,2)#torch.Size([batch, seqlen, 3]),a,t,v
        features_mask_del_umask = self.delete_umask(features_mask,umask)#torch.Size([umask_no_0, 3]),a,t,v
        a = self.delete_umask(a,umask)
        a_adj,a = self.get_affinity_matrix(a,self.adim,features_mask_del_umask[:,0])
        t = self.delete_umask(t,umask)
        t_adj,t = self.get_affinity_matrix(t,self.tdim,features_mask_del_umask[:,1])
        v = self.delete_umask(v,umask)
        v_adj,v = self.get_affinity_matrix(v,self.vdim,features_mask_del_umask[:,2])
        
        a,t,v,a_adj,t_adj,v_adj = self.knn_fill(a,t,v,a_adj,t_adj,v_adj,features_mask_del_umask)
        
        k1 = self.k1
        k2 = self.k2
        mu = self.mu
        
        F_a = self.MixedFilter(a, a_adj, t_adj, v_adj, k1, k2, mu)
        encoded_a = self.SelfFilter(F_a, a_adj, self.degree, self.alpha)
        encoded_a = self.Wa(encoded_a)
        featureInfo_a = torch.sigmoid(self.weight_a(encoded_a))
        encoded_a = encoded_a * featureInfo_a
        
        F_t = self.MixedFilter(t, t_adj, v_adj, a_adj,  k1, k2, mu)
        encoded_t = self.SelfFilter(F_t, t_adj, self.degree, self.alpha)
        encoded_t = self.Wt(encoded_t)
        featureInfo_t = torch.sigmoid(self.weight_t(encoded_t))
        encoded_t = encoded_t * featureInfo_t
        
        F_v = self.MixedFilter(v, v_adj, a_adj, t_adj, k1, k2, mu)
        encoded_v = self.SelfFilter(F_v, v_adj, self.degree, self.alpha)
        encoded_v = self.Wv(encoded_v)
        featureInfo_v = torch.sigmoid(self.weight_v(encoded_v))
        encoded_v = encoded_v * featureInfo_v
        
        featureInfo_loss = torch.mean(featureInfo_a) + torch.mean(featureInfo_t) + torch.mean(featureInfo_v)
        
        #encode:[batch, seqlen, dim]
        
        decoded_a = self.decoder_audio(encoded_a)
        decoded_t = self.decoder_text(encoded_t)
        decoded_v = self.decoder_video(encoded_v)
        
        
        decoded_a = self.add_umask(decoded_a,umask)
        decoded_a = decoded_a.view(raw_shape[0],raw_shape[1],-1)
        decoded_t = self.add_umask(decoded_t,umask)
        decoded_t = decoded_t.view(raw_shape[0],raw_shape[1],-1)
        decoded_v = self.add_umask(decoded_v,umask)
        decoded_v = decoded_v.view(raw_shape[0],raw_shape[1],-1)
        
        encoded_a = self.add_umask(encoded_a,umask)
        encoded_a = encoded_a.view(raw_shape[0],raw_shape[1],-1)
        encoded_t = self.add_umask(encoded_t,umask)
        encoded_t = encoded_t.view(raw_shape[0],raw_shape[1],-1)
        encoded_v = self.add_umask(encoded_v,umask)
        encoded_v = encoded_v.view(raw_shape[0],raw_shape[1],-1)
        
        hidden = torch.cat([encoded_a,encoded_t,encoded_v], dim=2).permute(1,0,2)#[batch, seqlen, dim]
        reconfiguration_result = [torch.cat([decoded_a,decoded_t,decoded_v], dim=2).permute(1,0,2)]
        
        return reconfiguration_result,hidden,[encoded_a,encoded_t,encoded_v,featureInfo_loss]


class IndexFlatL2:
    def __init__(self, vectors):
        self.vectors = torch.tensor(vectors)

    def search(self, query_vector, k):
        distances = torch.norm(self.vectors - query_vector, p=2, dim=1)
        if distances.shape[0] < k:
            k = distances.shape[0]
        nearest_distances, nearest_indices = torch.topk(distances, k, largest=False)
        return nearest_indices, nearest_distances