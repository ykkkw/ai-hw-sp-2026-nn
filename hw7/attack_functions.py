import torch
import torch.nn as nn

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def pgd_attack(model, images, labels, epsilon, alpha=2/255, iters=10):
    original_images = images.clone().detach()
    images = images.clone().detach() + torch.FloatTensor(*images.shape).uniform_(-epsilon, epsilon).to(images.device)
    images = torch.clamp(images, 0, 1).detach().requires_grad_(True)
    
    for _ in range(iters):
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach().requires_grad_(True)
        
    return images

def momentum_ifgsm_attack(model, images, labels, epsilon, alpha=2/255, iters=10, decay=1.0):
    original_images = images.clone().detach()
    images = images.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(images)
    
    for _ in range(iters):
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        gradients = images.grad
        gradients = gradients / torch.mean(torch.abs(gradients), dim=(1, 2, 3), keepdim=True)
        momentum = decay * momentum + gradients
        
        adv_images = images + alpha * momentum.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach().requires_grad_(True)
        
    return images