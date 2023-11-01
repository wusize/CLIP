import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import clip
from clip.clip import _transform

# from PIL import Image
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)


class CoOp(nn.Module):
    def __init__(self, clip_version="ViT-B/32", prompt_len=4):
        super().__init__()
        self.model, _ = clip.load(clip_version)

        for param in self.model.parameters():
            param.requires_grad = False    # freeze the CLIP model

        prompt_dim = self.model.positional_embedding.shape[1]

        # [p_0, p_k] + embed("A face of a man / woman")
        self.gender_prompt = nn.Parameter(torch.randn(2, prompt_len, prompt_dim))   # 0: male, 1: female
        # or self.gender_prompt = nn.Parameter(torch.randn(prompt_len, prompt_dim))

        # [p_0, p_k] + embed("A face of a person that is 24 years old")
        self.age_prompt = nn.Parameter(torch.randn(100, prompt_len, prompt_dim))   # 1 ~ 100 years old
        # or self.age_prompt = nn.Parameter(torch.randn(prompt_len, prompt_dim))

        # [p_0, p_k] + embed("A face of a man / woman that is 24 years old")
        self.gender_age_prompt = nn.Parameter(torch.randn(2 * 100, prompt_len, prompt_dim))

    @staticmethod
    def build_attention_mask(context_length):
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text_fast(self, text):    # a trick to speed up forward
        max_len = (text > 0).sum(-1).max().item()
        text = text[:, :max_len]
        x = self.model.token_embedding(text).type(self.model.dtype)  # [batch_size, max_len, d_model]
        x = x + self.model.positional_embedding[:max_len].type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x, self.build_attention_mask(max_len))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection

        return x

    def encode_text_with_prompt(self, text, prompt):
        # text: bs ctx_len
        max_len = (text > 0).sum(-1).max().item()
        text = text[:, :max_len]
        x = self.model.token_embedding(text).type(self.model.dtype)  # [batch_size, max_len, d_model]
        x = x + self.model.positional_embedding[:max_len].type(self.model.dtype)
        # x: bs max_len d_model
        # prompt: bs prompt_len d_model, learnable parameters
        prompt_len = prompt.shape[1]
        x = torch.cat([prompt, x], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x, self.build_attention_mask(max_len))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1) + prompt_len] @ self.model.text_projection

        return x

    def get_clip_text_embeddings(self):
        gender_texts = clip.tokenize(["A face of a man.", "A face of a woman."])
        age_texts = clip.tokenize([f"A face of a person that is {i + 1} years old." for i in range(100)])
        gender_age_texts = clip.tokenize([f"A face of a {gender} that is {i+1} years old."
                             for gender in ['man', 'woman'] for i in range(100)])

        gender_embeddings = self.encode_text_with_prompt(gender_texts, prompt=self.gender_prompt)  # 2 d_model

        age_embeddings = self.encode_text_with_prompt(age_texts, prompt=self.age_prompt)  # 100 d_model

        gender_age_embeddings = self.encode_text_with_prompt(gender_age_texts, prompt=self.gender_age_prompt)
        # 200 d_model

        # Do remember to normalize the last dim
        gender_embeddings = F.normalize(gender_embeddings, dim=-1)
        age_embeddings = F.normalize(age_embeddings, dim=-1)
        gender_age_embeddings = F.normalize(gender_age_embeddings, dim=-1)

        return gender_embeddings, age_embeddings, gender_age_embeddings

    def forward(self, image, labels=None):   # the image has been pre-processed, bs, 3, h, w
        with torch.no_grad():  # grad is not used for image branch
            image_embeddings = self.model.encode_image(image)  # bs d_model
            image_embeddings = F.normalize(image_embeddings, dim=-1)
        gender_embeddings, age_embeddings, gender_age_embeddings = self.get_clip_text_embeddings()
        logit_scale = self.model.logit_scale.exp()
        gender_logits = logit_scale * image_embeddings @ gender_embeddings.T
        age_logits = logit_scale * image_embeddings @ age_embeddings.T
        gender_age_logits = logit_scale * image_embeddings @ gender_age_embeddings.T

        if self.training:
            gender_labels = labels['gender']   # bs    {0: "man", 1: "woman"}
            age_labels = labels['age']    # bs        {0: 1, 1: 2, ...., 99: 100}
            gener_age_labels = gender_labels * 100 + age_labels

            # choice: 1
            gender_loss = F.cross_entropy(gender_logits, gender_labels)
            age_loss = F.cross_entropy(age_logits, age_labels)
            loss = gender_loss + age_loss

            # Choice: 2
            gener_age_loss = F.cross_entropy(gender_age_logits, gener_age_labels)
            loss = gener_age_loss

            # Choice: 3
            loss = gender_loss + age_loss + gener_age_loss
            return loss

        else:
            # choice: 1
            gender_pred = gender_logits.argmax(dim=-1)
            age_pred = gender_age_logits.argmax(dim=-1)

            # choice: 2
            gender_age_scores = gender_age_logits.softmax(dim=-1).view(2, 100)
            gender_scores = gender_age_scores.sum(-1)
            age_scores = gender_age_scores.sum(0)

            gender_pred = gender_scores.argmax(dim=-1)
            age_pred = age_scores.argmax(dim=-1)

            return gender_pred, age_pred


class YourDataset(Dataset):
    def __init__(self, image_size=224):
        super().__init__()
        self.transform = _transform(image_size)   # image processor

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
