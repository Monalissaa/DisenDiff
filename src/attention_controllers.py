import abc
import torch
LOW_RESOURCE = False
import math


class CDAttentionStore(abc.ABC):

    @staticmethod
    def get_empty_store():
        return {8: [], 16: [], 32: [], 64: []}
        # return {"down_cross": [], "mid_cross": [], "up_cross": [],
        #         "down_self": [],  "mid_self": [],  "up_self": []}

    # def forward(self, attn, is_cross: bool = True, place_in_unet: str = None):
    def __call__(self, attn, is_cross: bool = True, place_in_unet: str = None):
        # key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        key = math.sqrt(attn.shape[1])
        self.step_store[key].append(attn)
        return
    
    

    def between_steps(self):
        # if len(self.attention_store) == 0:
        #     self.attention_store = self.step_store
        # else:
        #     for key in self.attention_store:
        #         for i in range(len(self.attention_store[key])):
        #             self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    # def get_average_attention(self):
    #     # average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
    #     average_attention = {key: [item for item in self.attention_store[key]] for key in self.attention_store}

    #     return average_attention


    def reset(self):
        # super(CDAttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        # self.attention_store = {}

    def __init__(self):
        # super(CDAttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        # self.attention_store = {}
# #############################################################################################################
# class AttentionControl(abc.ABC):
    
#     def step_callback(self, x_t):
#         return x_t
    
#     def between_steps(self):
#         return
    
#     @property
#     def num_uncond_att_layers(self):
#         return self.num_att_layers if LOW_RESOURCE else 0
    
#     @abc.abstractmethod
#     def forward(self, attn, is_cross: bool, place_in_unet: str):
#         raise NotImplementedError

#     def __call__(self, attn, is_cross: bool, place_in_unet: str):
#         if self.cur_att_layer >= self.num_uncond_att_layers:
#             if LOW_RESOURCE:
#                 attn = self.forward(attn, is_cross, place_in_unet)
#             else:
#                 h = attn.shape[0]
#                 attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
#         self.cur_att_layer += 1
#         if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
#             self.cur_att_layer = 0
#             self.cur_step += 1
#             self.between_steps()
#         return attn
    
#     def reset(self):
#         self.cur_step = 0
#         self.cur_att_layer = 0

#     def __init__(self):
#         self.cur_step = 0
#         self.num_att_layers = -1
#         self.cur_att_layer = 0


# class AttentionStore(AttentionControl):

#     @staticmethod
#     def get_empty_store():
#         return {"down_cross": [], "mid_cross": [], "up_cross": [],
#                 "down_self": [],  "mid_self": [],  "up_self": []}

#     def forward(self, attn, is_cross: bool, place_in_unet: str):
#         key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
#         if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
#             self.step_store[key].append(attn)
#         return attn

#     def between_steps(self):
#         if len(self.attention_store) == 0:
#             self.attention_store = self.step_store
#         else:
#             for key in self.attention_store:
#                 for i in range(len(self.attention_store[key])):
#                     self.attention_store[key][i] += self.step_store[key][i]
#         self.step_store = self.get_empty_store()

#     def get_average_attention(self):
#         average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
#         return average_attention


#     def reset(self):
#         super(AttentionStore, self).reset()
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}

#     def __init__(self):
#         super(AttentionStore, self).__init__()
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}



##############################################################################################################


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        # print(self.num_att_layers)
        # print(self.num_uncond_att_layers)
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # print(self.cur_step)
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:  # avoid memory overhead  origin:32 ** 2
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
            if self.all_step_attention_store:
                self.all_step_attention[self.cur_step-1] = {"down_cross": [], "mid_cross": [], "up_cross": [],
                    "down_self": [],  "mid_self": [],  "up_self": []}
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.all_step_attention[self.cur_step-1][key].append(self.step_store[key][i].clone())
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
            
            if self.all_step_attention_store:
                self.all_step_attention[self.cur_step-1] = self.step_store
        
        # 
        self.step_store = self.get_empty_store()
        

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, all_step_attention_store=False):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.all_step_attention_store = all_step_attention_store
        if all_step_attention_store:
            self.all_step_attention = {}

