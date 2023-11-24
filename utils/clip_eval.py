import clip
import torch
from torchvision import transforms
from tqdm import tqdm

# from ldm.models.diffusion.ddim import DDIMSampler

def replace_chars(string):    
    string = string.replace(" ", "_").replace("<","").replace(">","")
    return string

class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


# class LDMCLIPEvaluator(CLIPEvaluator):
#     def __init__(self, device, clip_model='ViT-B/32') -> None:
#         super().__init__(device, clip_model)

#     def evaluate(self, ldm_model, src_images, target_text, n_samples=64, n_steps=50):
        
#         sampler = DDIMSampler(ldm_model)

#         samples_per_batch = 8
#         n_batches         = n_samples // samples_per_batch

#         # generate samples
#         all_samples=list()
#         with torch.no_grad():
#             with ldm_model.ema_scope():                
#                 uc = ldm_model.get_learned_conditioning(samples_per_batch * [""])

#                 for batch in range(n_batches):
#                     c = ldm_model.get_learned_conditioning(samples_per_batch * [target_text])
#                     shape = [4, 256//8, 256//8]
#                     samples_ddim, _ = sampler.sample(S=n_steps,
#                                                     conditioning=c,
#                                                     batch_size=samples_per_batch,
#                                                     shape=shape,
#                                                     verbose=False,
#                                                     unconditional_guidance_scale=5.0,
#                                                     unconditional_conditioning=uc,
#                                                     eta=0.0)

#                     x_samples_ddim = ldm_model.decode_first_stage(samples_ddim)
#                     x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)

#                     all_samples.append(x_samples_ddim)
        
#         all_samples = torch.cat(all_samples, axis=0)

#         sim_samples_to_img  = self.img_to_img_similarity(src_images, all_samples)
#         sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), all_samples)

#         return sim_samples_to_img, sim_samples_to_text
import re
class ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, gen_samples, src_images, target_text):

        sim_samples_to_img  = self.img_to_img_similarity(src_images, gen_samples)
        # sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), gen_samples)
        clean_text = re.sub(r"<new\d+>", " ", target_text).strip()
        sim_samples_to_text = self.txt_to_img_similarity(clean_text, gen_samples)
        # sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("<new1>", ""), gen_samples)

        return sim_samples_to_img, sim_samples_to_text

import argparse
from ldmm.data.personalized import PersonalizedBase
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--real_data_dir",
        type=str,
        default=None,
        required=True,
    )

    parser.add_argument(
        "--prompts_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--fake_data_dir",
        type=str,
        default=None,
        required=True,
    )

    parser.add_argument(
        "--image_align_only_training_prompt",
        default=False,
        action="store_true",
        # required=True,
    )

    parser.add_argument(
        "--text_align_wo_training_prompt",
        default=False,
        action="store_true",
        # required=True,
    )
    

    args = parser.parse_args()
    return args

args = parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

evaluator = ImageDirEvaluator(device)

real_data_dir = args.real_data_dir
real_data_loader = PersonalizedBase(real_data_dir, size=512, flip_p=0.0)
real_images = [torch.from_numpy(real_data_loader[i]["image"]).permute(2, 0, 1) for i in range(real_data_loader.num_images)]

real_images = torch.stack(real_images, axis=0)

prompts_path = args.prompts_path
prompts = []
with open(prompts_path) as f:
    for line in f:
        prompts.append(line.strip())

if args.image_align_only_training_prompt:
    prompts = [prompts[0]]
elif args.text_align_wo_training_prompt:
    prompts = prompts[1:]

sim_img_list, sim_text_list = [], []
for prompt in tqdm(prompts):
    save_prompt_name = replace_chars(prompt)

    fake_data_dir = os.path.join(args.fake_data_dir, save_prompt_name)
    fake_data_loader = PersonalizedBase(fake_data_dir, size=512, flip_p=0.0)
    fake_images = [torch.from_numpy(fake_data_loader[i]["image"]).permute(2, 0, 1) for i in range(fake_data_loader.num_images)]
    fake_images = torch.stack(fake_images, axis=0)

    # prompt = 'cat swimming in a pool'

    sim_img, sim_text = evaluator.evaluate(fake_images, real_images, prompt)
    sim_img_list.append(sim_img.item())
    sim_text_list.append(sim_text.item())

print("Image similarity: ", sum(sim_img_list)/len(sim_img_list))
print("Text similarity: ", sum(sim_text_list)/len(sim_text_list))

with open(os.path.join(args.fake_data_dir, "text_alignment.txt"), "a+") as file:
    if args.image_align_only_training_prompt:
        file.write("Training Prompt Image similarity: {}\n".format(sum(sim_img_list)/len(sim_img_list)))
        file.write("Training Prompt Text similarity: {}\n".format(sum(sim_text_list)/len(sim_text_list)))
    elif args.text_align_wo_training_prompt:
        file.write("(WO Training Prompt) Image similarity: {}\n".format(sum(sim_img_list)/len(sim_img_list)))
        file.write("(WO Training Prompt) Text similarity: {}\n".format(sum(sim_text_list)/len(sim_text_list)))
    else:
        file.write("Image similarity: {}\n".format(sum(sim_img_list)/len(sim_img_list)))
        file.write("Text similarity: {}\n".format(sum(sim_text_list)/len(sim_text_list)))