import torch
import torch.nn as nn
from collections import deque
from mol_tree import Vocab, MolTree
from nnutils import create_var, GRU

MAX_NB = 8

class JTNNEncoder(nn.Module):

    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, root_batch):
        orders = []
        for root in root_batch:
            order = get_prop_order(root)
            orders.append(order)
        
        h = {}
        max_depth = max([len(x) for x in orders])
        padding = create_var(torch.zeros(self.hidden_size), False)

        for t in xrange(max_depth):  # clique(tree의 노드)별로 수행 
            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            cur_x = []
            cur_h_nei = []
            for node_x,node_y in prop_list: ## prop_list # 
                                            

                x,y = node_x.idx,node_y.idx  ## = [[A,G], [G,E], [E, H], ... ...] 
                cur_x.append(node_x.wid)     ## node_x.wid : ID of each node

                h_nei = []
                for node_z in node_x.neighbors:  # 해당 for문은 하나의 노드에서 연결된 노드(1개 이상)를 연결하는 역할을 함  
                    z = node_z.idx
                    if z == y: continue
                    h_nei.append(h[(z,x)])

                pad_len = MAX_NB - len(h_nei) # 대신 하나의 노드에서 연결된 엣지의 수는 8개 이하로 제한
                h_nei.extend([padding] * pad_len) # 8개가 되지 않은 나머지는 0으로 padding
                
                
                cur_h_nei.extend(h_nei) # 위 과정을 트리의 속해 있는 노드의 이웃된 노드만큼 반복
                                        # 그러므로 "cur_h_nei" 각 노드 별로 이웃하는 노드가 뭔지
                                        # 확인할 수 있음
                                        # GRU 관점에서 봤을 때 old memory를 의미함 (??)


            cur_x = create_var(torch.LongTensor(cur_x)) # GRU에서 X값으로 생각하면 됨
            cur_x = self.embedding(cur_x)               # GRU에서 Input으로 하기 위해 X value를
                                                        # embedding을 해주는 과정
                                                        
                                                        # 이 때의 cur_x의 size는 
                                                        # [batch_size, embedding_size] = [40, 450] 


            cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
            # 이 떄의 cur_h_nei의 size는 [batch_size, max_neighbors(=8), embedding_size] = [40, 8 ,450]


            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            # 하나의 노드와 그 노드와 연결된 노드들의 정보들이 GRU가 학습됨
            # new_h의 size : [batch_size, max_neighbors(=8), embedding_size]

            for i,m in enumerate(prop_list):
                
                x,y = m[0].idx,m[1].idx
                 # (x, y) = = [[A,G], [G,E], [E, H], ... ...] 
                h[(x,y)] = new_h[i] # size : [BATCH_SIZE, 450]
                                    # 실제 PRINT문을 보면 size=[450]으로 나옴... 왜그럴까...
        root_vecs = node_aggregate(root_batch, h, self.embedding, self.W)
        # 루트 노드의 최종 output 값을 산출 (루트 노드???)
        # 이를 통해 latent_space(tree) get
        # root_vecs size : [40, 450]
        return h, root_vecs

def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    # x-size : [batch_size, embedding_size] = [40, 450] 
    # h_nei-size : [batch_size, max_neighbors(=8), embedding_size] = [40, 8 ,450]

    hidden_size = x.size()[-1]
    sum_h = h_nei.sum(dim=1) # size : [40, 450]
    z_input = torch.cat([x,sum_h], dim=1)
    z = nn.Sigmoid()(W_z(z_input))

    r_1 = W_r(x).view(-1,1,hidden_size)
    r_2 = U_r(h_nei)
    r = nn.Sigmoid()(r_1 + r_2)
    
    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x,sum_gated_h], dim=1)
    pre_h = nn.Tanh()(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h # size : [40, 450]
"""
Helper functions
"""

def get_prop_order(root):
    queue = deque([root])
    visited = set([root.idx])
    root.depth = 0
    order1,order2 = [],[]
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth-1].append( (x,y) )
                order2[y.depth-1].append( (y,x) )
    order = order2[::-1] + order1
    return order

def node_aggregate(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)
    
    h_nei = torch.cat(h_nei, dim=0).view(-1,MAX_NB,hidden_size)
    sum_h_nei = h_nei.sum(dim=1)
    x_vec = create_var(torch.LongTensor(x_idx))
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)
    return nn.ReLU()(W(node_vec))
