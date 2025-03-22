import torch
import torch.nn as nn
from .neuron_astrocyte import *

class ProliferationLayer(nn.Module):
    """ Neurogenesis phase 1: Proliferation """
    
    def __init__(self, block, custom_layer, attention_layer):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, output_dim)
        self.layer_num = 1
        self.nhead = 5
        self.block_layer = block
        self.softmax = nn.Softmax(dim=1)
        self.custom_layer = custom_layer
        self.custom_self_attn = attention_layer

    def forward(self, x):
 
        #instantiate custom attention
        # custom_self_attn = AlbertAttention(config = model.config)

        self.custom_self_attn.key = self.block_layer.attention.key
        self.custom_self_attn.value = self.block_layer.attention.value
        self.custom_self_attn.query = self.block_layer.attention.query
        self.custom_self_attn.attention_dropout = self.block_layer.attention.attention_dropout
        self.custom_self_attn.output_dropout = self.block_layer.attention.output_dropout
        self.custom_self_attn.dense = self.block_layer.attention.dense



        #instantiate custom layer
        # custom_layer = AlbertLayer(config = model.config)
        self.custom_layer.attention = self.custom_self_attn
        self.custom_layer.ffn = self.block_layer.ffn
        self.custom_layer.ffn_output = self.block_layer.ffn_output


        outputs_of_previous_layer = output['hidden_states'][self.layer_num-1]
        custom_outputs, attention_scores, attention_probs, query_layer, key_layer = self.custom_layer(outputs_of_previous_layer)


        #normalize Q and K matrices appropriately
        query_layer /= custom_self_attn.attention_head_size**(1/4)
        key_layer /= custom_self_attn.attention_head_size**(1/4)

        query_layer = query_layer.detach().numpy()
        key_layer = key_layer.detach().numpy()

        #Constructing phi (random features)

        phi_low_m = get_phi(m = int(1e3), D = custom_self_attn.attention_head_size,which_phi = 'performer')
        phi_high_m = get_phi(m = int(1e5), D = custom_self_attn.attention_head_size,which_phi = 'performer')

        astro_ps_low_m = get_astro_responses(query_layer,key_layer, self.nhead,phi_low_m)

        astro_ps_high_m = get_astro_responses(query_layer,key_layer, self.nhead,phi_high_m)

        attention_normalizations = torch.exp(attention_scores[0,self.nhead]).sum(-1).detach().numpy()

        return  astro_ps_high_m, astro_ps_low_m, attention_normalizations


    def astrocyte_process(self):
        """ apply astrocyte activation function """
        pass
    def microglia_process(self):
        """ apply microglia activation function """
        pass
    def oligodendrocyte_process(self):
        """ apply oligodendrocyte activation function """
        pass
    def save(self, path):
        torch.save(self.state_dict(), path)