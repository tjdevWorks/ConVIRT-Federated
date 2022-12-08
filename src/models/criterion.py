import torch
import torchmetrics

class ConVIRTContrastiveCriterion(torch.nn.Module):
    def __init__(self, temperature: float, lamda:float):
        super().__init__()
        self.temperature = temperature
        self.lamda = lamda
    
    def forward(self, image_v, text_u):
        img_text_sim = torchmetrics.functional.pairwise_cosine_similarity(image_v, text_u)

        text_img_sim = torchmetrics.functional.pairwise_cosine_similarity(text_u, image_v)

        l_vu = torch.diag(-torch.nn.functional.log_softmax(img_text_sim / self.temperature, 1))
        l_uv = torch.diag(-torch.nn.functional.log_softmax(text_img_sim / self.temperature, 1))

        loss = self.lamda * l_vu + (1-self.lamda)*l_uv

        mean_loss = torch.mean(loss)
        return mean_loss