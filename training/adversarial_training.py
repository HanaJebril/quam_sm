import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Callable
import torch.nn.functional as F




def morph_op(mask, op="dilate", kernel_size=3, iterations=1):
    """
    Morphological operation with configurable iterations.
    mask: shape [1,H,W]
    """
    out = mask.clone()
    for _ in range(iterations):
        out = out.float().unsqueeze(0)  # [B=1,1,H,W]
        kernel = torch.ones((1,1,kernel_size,kernel_size), device=mask.device)
        padding = kernel_size // 2

        if op == "dilate":
            out = F.conv2d(out, kernel, padding=padding)
            out = (out > 0).float()
        elif op == "erode":
            out = F.conv2d(out, kernel, padding=padding)
            out = (out == kernel.numel()).float()
        else:
            raise ValueError("op must be 'dilate' or 'erode'")

        out = out.squeeze(0)  # [1,H,W]
    return out.long()



def make_target_mask(strategy: str, step: int, H: int, W: int, device: str):
    """Generate synthetic target mask for adversarial training."""
    if strategy == 'zero':
        return torch.zeros((1, H, W), dtype=torch.long, device=device)
    elif strategy == 'one':
        return torch.ones((1, H, W), dtype=torch.long, device=device)
    elif strategy == 'alternating':
        return torch.ones((1, H, W), dtype=torch.long, device=device) if step % 2 == 0 \
               else torch.zeros((1, H, W), dtype=torch.long, device=device)
    elif strategy == 'random':
        return torch.randint(0, 2, (1, H, W), dtype=torch.long, device=device)
    else:
        raise ValueError(f"Unknown target strategy: {strategy}")


def train_adversarial_model_segmentation(
    ref_model,
    train_loader,
    test_image,
    idx_model: int,
    save_dir: str,
    num_epochs: int = 5,
    c_scheduling: Callable[[int], float] = lambda step: 1 * 2 ** step,
    penalties_per_optimisation_step: int = 80,
    initial_c: float = 1.0,
    gamma: float = 0.05,
    lr: float = 5e-3,
    device: str = "cuda",
    target_strategy: str = "alternating"
):
    """
    Train a single adversarial segmentation model with QUAM-style penalties.
    Returns intermediate predictions and loss traces.
    """
    store_device = "cpu"
    loss_ce = nn.CrossEntropyLoss()

    ref_model.eval()
    ref_model = ref_model.to(device)
    test_image = test_image.to(device)

    n_points, C, H, W = test_image.shape
    n_classes = 2

    # Storage
    n_penalties = len(train_loader) // penalties_per_optimisation_step
    test_pt_preds = torch.zeros((n_points, num_epochs, n_penalties, n_classes, H, W),
                                dtype=torch.float32, device=device)
    test_best_pred = torch.zeros((n_points, n_classes, H, W),
                                 dtype=torch.float32, device=device)
    opt_losses = torch.zeros((n_points, num_epochs, n_penalties, n_classes),
                             dtype=torch.float32, device=device)
    pen_losses = torch.zeros((num_epochs, n_penalties, n_classes),
                             dtype=torch.float32, device=device)
    train_losses = torch.zeros((num_epochs, n_penalties, n_classes),
                               dtype=torch.float32, device=device)

    # Reference training losses
    with torch.no_grad():
        model_train_loss = torch.zeros((len(train_loader), 1), dtype=torch.float32, device=device)
        for ii, (images, masks, _) in enumerate(tqdm(train_loader, desc="Reference Model Loss")):
#             first_mask = masks[0:1].to(device)
            images, masks = images.to(device), masks.to(device)
            preds = ref_model(images)
            model_train_loss[ii, :] = loss_ce(preds, masks)

    # Initial reference prediction
    with torch.no_grad():
        average_net_pred = torch.softmax(ref_model(test_image), dim=1).detach()
        base_mask = torch.argmax(average_net_pred, dim=1, keepdim=True)  # [N,1,H,W]
        first_mask = base_mask[0].to(device)  

    if idx_model % 2 == 0:
        tgt_mask = morph_op(first_mask, op="dilate", kernel_size=3, iterations=3)
    else:
        tgt_mask = morph_op(first_mask, op="erode", kernel_size=3, iterations=3)

    # Init adversarial model
    adv_model = copy.deepcopy(ref_model).to(device)
    optimizer = optim.Adam(adv_model.parameters(), lr=lr)

    best_model = None
    best_div = -float("inf")
    c = initial_c
    global_step = 0
    adv_model.train()
    # === Training loop ===
    for epoch in range(num_epochs):
#         losses = torch.zeros(n_classes, dtype=torch.float32, device=device)
        

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for idx, (images, masks, _) in enumerate(loop):
            images, masks = images.to(device), masks.to(device)


            # === Penalty update step ===
            if (idx + 1) % penalties_per_optimisation_step == 0:
                cur_penalty = ((idx + 1) // penalties_per_optimisation_step) - 1
#                 tgt_mask = make_target_mask(target_strategy, idx_model, H, W, device)

                    
                if epoch == 0 and cur_penalty == 0: 
                    plt.figure(figsize=(8,4))

                    plt.subplot(1,2,1)
                    plt.imshow(first_mask.squeeze().cpu().numpy(), cmap="gray")
                    plt.title("Original First Mask")
                    plt.axis("off")

                    plt.subplot(1,2,2)
                    plt.imshow(tgt_mask.squeeze().cpu().numpy(), cmap="gray")
                    plt.title("Target Mask (Dilate/Erode)")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"Mask_Comparison_{idx_model}.png"))
                    plt.close()

                
                
                
                
                adv_preds = adv_model(images)
                train_loss = loss_ce(adv_preds, masks)


                pen = c * (train_loss - model_train_loss[idx].detach() - gamma)
                c = c_scheduling(epoch * len(train_loader) + idx)
 
                adv_test_pred = adv_model(test_image)
                loss_adv = loss_ce(adv_test_pred, tgt_mask)
                total_loss = loss_adv + pen.mean()


                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                global_step += 1
                with torch.no_grad():
#                         adv_model.eval()
                    train_losses[epoch, cur_penalty, :] = train_loss.detach().to(store_device)
                    test_pred_soft = torch.softmax(adv_model(test_image), dim=1).cpu()
                    test_pt_preds[:, epoch, cur_penalty, :, :, :] = test_pred_soft
                    pen_losses[epoch, cur_penalty, :] = pen.detach().cpu()
                    opt_losses[:, epoch, cur_penalty, :] = loss_adv.detach().cpu()  
#                     losses = torch.zeros_like(losses)
                    if loss_adv.item() > best_div and pen <= 0:
                        best_div = loss_adv.item()
                        best_model = copy.deepcopy(adv_model)
                        test_best_pred[:, :, :, :] = test_pred_soft

#                         adv_model.train()



    # Flatten storage for downstream usage
    pen_losses = pen_losses.flatten(0, 1).unsqueeze(0).expand(n_points, -1, -1)
    opt_losses = opt_losses.flatten(1, 2)
    train_losses = train_losses.flatten(0, 1).unsqueeze(0).expand(n_points, -1, -1)
    test_pt_preds = test_pt_preds.flatten(1, 2)
    
    
    
    
    
    
        # Move tensors to CPU
    train_losses_np = train_losses[0].cpu().numpy()  # shape: [len(train_dl) * n_epochs, n_classes]
    pen_losses_np = pen_losses[0].cpu().numpy()      # shape: [n_penalties * n_epochs, n_classes]
    opt_losses_np = opt_losses[:, :, :].cpu().numpy()  # shape: [n_points, n_penalties * n_epochs, n_classes]
    
    
    # Plot training loss per epoch
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses_np[:, 0])
    plt.title("Train Loss over Batches × Epochs")
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"Train_{idx_model}.png")
    plt.savefig(save_path)
    plt.close()

    # Plot penalty loss per epoch
    plt.figure(figsize=(10, 4))
    plt.plot(pen_losses_np[:, 0])
    plt.title("Penalty Loss per Penalty Step")
    plt.xlabel("Penalty Step Index")
    plt.ylabel("Penalty Loss")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"Penalty_{idx_model}.png")
    plt.savefig(save_path)
    plt.close()

    # Plot optimization loss on test point
    plt.figure(figsize=(10, 4))
    plt.plot(opt_losses_np[0, :, 0])
    plt.title("Optimization Loss on Test Point")
    plt.xlabel("Penalty Step Index")
    plt.ylabel("Test Point Loss")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"Optimization_{idx_model}.png")
    plt.savefig(save_path)
    plt.close()
    
    

    return {
        "average_net_pred": average_net_pred,
        "sample_preds": test_pt_preds,
        "model_train_loss": model_train_loss,
        "train_loss": train_losses,
        "pen_loss": pen_losses,
        "obj_loss": opt_losses,
        "test_best_pred": test_best_pred,
    }


def generate_adversarial_models_seg(
    ref_model, train_loader, test_image,
    save_dir: str, num_models: int = 5, **kwargs
):
    """
    Train multiple adversarial models and collect predictions/losses.
    """
    preds_all, obj_loss_all, pen_loss_all, train_loss_all, test_best_pred_all = None, None, None, None, None

    for m in range(num_models):
        print(f"▶ Training adversarial model {m+1}/{num_models}")
        out = train_adversarial_model_segmentation(
            ref_model, train_loader, test_image,
            idx_model=m, save_dir=save_dir, **kwargs
        )

        sample_preds = out["sample_preds"].detach()
        obj_loss = out["obj_loss"].detach()
        pen_loss = out["pen_loss"].detach()
        train_loss = out["train_loss"].detach()
        test_best_pred = out["test_best_pred"].detach()

        if preds_all is None:
            n_points, n_samples, n_classes, H, W = sample_preds.shape
            preds_all = torch.zeros((n_points, n_samples, num_models, n_classes, H, W),
                                    dtype=sample_preds.dtype, device=sample_preds.device)
            obj_loss_all = torch.zeros((n_points, n_samples, num_models, n_classes),
                                       dtype=obj_loss.dtype, device=obj_loss.device)
            pen_loss_all = torch.zeros((n_points, n_samples, num_models, n_classes),
                                       dtype=pen_loss.dtype, device=pen_loss.device)
            train_loss_all = torch.zeros((n_points, train_loss.shape[1], num_models, train_loss.shape[2]),
                                         dtype=train_loss.dtype, device=train_loss.device)
            test_best_pred_all = torch.zeros((n_points, num_models, n_classes, H, W),
                                             dtype=test_best_pred.dtype, device=test_best_pred.device)

        preds_all[:, :, m, :, :, :] = sample_preds
        obj_loss_all[:, :, m, :] = obj_loss
        pen_loss_all[:, :, m, :] = pen_loss
        train_loss_all[:, :, m, :] = train_loss
        test_best_pred_all[:, m, :, :, :] = test_best_pred

    return {
        "average_net_pred": out["average_net_pred"],
        "sample_preds": preds_all,
        "model_train_loss": out["model_train_loss"],
        "train_loss": train_loss_all,
        "pen_loss": pen_loss_all,
        "obj_loss": obj_loss_all,
        "test_best_pred": test_best_pred_all,
    }
