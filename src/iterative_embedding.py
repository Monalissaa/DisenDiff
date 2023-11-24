from diffusers.models.embeddings import TimestepEmbedding, Timesteps
import torch
class IterativeEmbedding(torch.nn.Module):
    def __init__(self, size, newtoken=1):
        super().__init__()
        
        timestep_input_dim = 768
        time_embed_dim = 2048
        flip_sin_to_cos = True
        freq_shift = 0
        self.time_proj = Timesteps(timestep_input_dim, flip_sin_to_cos, freq_shift)
        # timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn="silu",
            out_dim=timestep_input_dim,
            post_act_fn=None,
            cond_proj_dim=None,
        )
        # self.time_emb_proj = torch.nn.Linear(time_embed_dim, timestep_input_dim)
        
        # embed_dim = 768

        # self.iterative_embedding = torch.nn.Parameter(torch.randn(768), requires_grad=True)
        self.expand_embeddings = torch.nn.Embedding(size, 768)
        self.expand_embeddings.weight.data.zero_()
        self.expand_embeddings.weight.data[-newtoken:] = torch.ones(768)
        self.expand_embeddings.weight.requires_grad = False


    def forward(
        self,
        input_ids,
        timesteps,
    ) -> torch.Tensor:
        intput_shape = input_ids.size()

        timesteps = timesteps.expand(intput_shape[0])
        t_emb = self.time_proj(timesteps)


        # emb = self.time_embedding(t_emb)[:, :, None]
        emb = self.time_embedding(t_emb)
        # emb = emb.repeat(intput_shape[1]).view(intput_shape[0], intput_shape[1], -1)
        emb = emb.repeat(1, 77, 1).view(intput_shape[0], intput_shape[1], -1)
        # print(offset_embedding.shape)
        position_embedding = self.expand_embeddings(input_ids)
        embedding = emb * position_embedding
        return embedding
