import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from attack_functions import fgsm_attack, pgd_attack, momentum_ifgsm_attack

# same trained model from last best saved cnn model
class CNN(nn.Module):
    def __init__(self, 
                 inchannel = 1,
                 conv_channel = [32, 64, 128],
                 kernel_size = 3,
                 conv_dropout = 0.25,
                 fc_size = 256,
                 fc_dropout = 0.5,
                 output_size = 10,
                 ):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, conv_channel[0], kernel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(conv_channel[0], conv_channel[1], kernel_size)
        self.relu2 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(conv_dropout)

        self.conv3 = nn.Conv2d(conv_channel[1], conv_channel[2], kernel_size)
        self.relu3 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(conv_dropout)

        fc_input = conv_channel[2] * 5 * 5
        self.fc1  = nn.Linear(fc_input, fc_size)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(fc_dropout)
        self.fc2  = nn.Linear(fc_size, output_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pooling1(out)
        out = self.dropout1(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pooling2(out)
        out = self.dropout2(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        return out

class Embedding(nn.Module):
    def __init__(self, patch_height=4, img_width=28, d_model=128):   
        super(Embedding, self).__init__()
        self.patch_height = patch_height
        self.proj = nn.Linear(patch_height * img_width, d_model)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.unfold(1, self.patch_height, self.patch_height)
        x = x.reshape(x.size(0), x.size(1), -1)
        return self.proj(x)

def run_evaluation(model_path, epsilon=0.15, batch_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting compute device: {device}")
    
    model = CNN().to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError:
            model = torch.load(model_path, map_location=device)
    else:
        raise FileNotFoundError(f"Missing weight checkpoints at target: {model_path}")
    
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total, correct_before = 0, 0
    fgsm_fooled, pgd_fooled, m_ifgsm_fooled = 0, 0, 0

    print("Starting noise generation loops...")

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        correct_mask = (preds == labels)
        correct_before += correct_mask.sum().item()
        total += labels.size(0)
        
        if correct_mask.sum() == 0:
            continue
            
        clean_images = images[correct_mask]
        clean_labels = labels[correct_mask]
        
        # FGSM
        fgsm_inputs = clean_images.clone().detach().requires_grad_(True)
        fgsm_outputs = model(fgsm_inputs)
        fgsm_loss = nn.CrossEntropyLoss()(fgsm_outputs, clean_labels)
        model.zero_grad()
        fgsm_loss.backward()
        
        adv_fgsm = fgsm_attack(fgsm_inputs, epsilon, fgsm_inputs.grad)
        with torch.no_grad():
            fgsm_fooled += (torch.max(model(adv_fgsm), 1)[1] != clean_labels).sum().item()

        # PGD
        adv_pgd = pgd_attack(model, clean_images, clean_labels, epsilon, alpha=2/255, iters=20)
        with torch.no_grad():
            pgd_fooled += (torch.max(model(adv_pgd), 1)[1] != clean_labels).sum().item()

        # Momentum I-FGSM
        adv_m_ifgsm = momentum_ifgsm_attack(model, clean_images, clean_labels, epsilon, alpha=2/255, iters=20)
        with torch.no_grad():
            m_ifgsm_fooled += (torch.max(model(adv_m_ifgsm), 1)[1] != clean_labels).sum().item()
            
    rec_rate_before = (correct_before / total) * 100
    asr_fgsm = (fgsm_fooled / correct_before) * 100
    asr_pgd = (pgd_fooled / correct_before) * 100
    asr_m_ifgsm = (m_ifgsm_fooled / correct_before) * 100

    print("\n" + "="*50)
    print("ATTACK COMPARATIVE METRICS")
    print("="*50)
    print(f"Model Path           : {model_path}")
    print(f"Recognition Rate    : {rec_rate_before:.2f}%")
    print("-"*50)
    print(f"FGSM Attack Success Rate    : {asr_fgsm:.2f}%")
    print(f"PGD Success Rate    : {asr_pgd:.2f}%")
    print(f"Momentum I-FGSM Success Rate: {asr_m_ifgsm:.2f}%")
    print("="*50)

if __name__ == "__main__":
    MODEL_PATH = "/Users/kk/Desktop/Sofia-MSCS/sp26/AI/hw4/cnn_best.pt"
    run_evaluation(model_path=MODEL_PATH, epsilon=0.15)