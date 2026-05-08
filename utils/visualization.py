import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from matplotlib.colors import ListedColormap

def plot_and_save_images(test_images, test_annotations, target_np, output_np, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(test_images.cpu()[0][0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(test_annotations.cpu()[0], cmap='gray')
    axes[1].set_title('Original Label')
    axes[1].axis('off')

    axes[2].imshow(target_np.cpu(), cmap='gray')
    axes[2].set_title('Segmentation Output')
    axes[2].axis('off')

    axes[3].imshow(output_np, cmap='gray')
    axes[3].set_title('Pseudo Label')
    axes[3].axis('off')

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base, ext = os.path.splitext(save_path)
    save_path = base + ".png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    mean_pred_folder = os.path.join(os.path.dirname(save_path), "Pseudo_label")
    os.makedirs(mean_pred_folder, exist_ok=True)
    mean_pred_fname = os.path.join(mean_pred_folder, os.path.basename(save_path))

    pred_img_uint8 = (output_np * 255).astype(np.uint8)
    im = Image.fromarray(pred_img_uint8, mode='L')
    im.save(mean_pred_fname)

# def plot_segmentation_results_final(test_images, test_annotations, out, epistemic,aleatoric, total , name, save_path):
    
#     # Ensure the save path directory exists
#     os.makedirs(save_path, exist_ok=True)

#     # Extract filename (without extension)
#     figname = name[0].split('.')[0]

#     # Create folders for saving individual images
#     folders = {
#         "original": os.path.join(save_path, "original_image"),
#         "ground_truth": os.path.join(save_path, "ground_truth"),
#         "mean_pred": os.path.join(save_path, "mean_prediction"),
#         "epistemic": os.path.join(save_path, "epistemic"),
#         "aleatoric": os.path.join(save_path, "aleatoric"),
#         "total": os.path.join(save_path, "total"),
# #         "pseudo_label": os.path.join(save_path, "pseudo_label")
#     }
    
#     for folder in folders.values():
#         os.makedirs(folder, exist_ok=True)  # Create each directory if it does not exist

#     # Create a subplot with 1 row and 5 columns
#     fig, axes = plt.subplots(1, 6, figsize=(24, 4))

#     # Define image data and titles
#     images_data = {
#         "original": test_images.cpu()[0].numpy(),
#         "ground_truth": np.mean(test_annotations.cpu()[0].numpy(), axis=0),
#         "mean_pred": out["pred_mean_argmax"][0, 0],
#         "epistemic": epistemic[0].cpu().numpy(),
#         "aleatoric": aleatoric[0].cpu().numpy(),
#         "total": total[0].cpu().numpy()
#     }

#     titles = {
#         "original": "Original Image",
#         "ground_truth": "Ground Truth",
#         "mean_pred": "Mean Model Predictions",
#         "epistemic": "Model Epistemic Uncertainty",
#         "aleatoric": "Model Aleatoric Uncertainty",
#         "total": "Model Total Uncertainty"
#     }

#     colormaps = {
#         "epistemic": "jet",  # Apply colormap for variance visualization
#         "aleatoric": "jet",
#         "total": "jet",
#     }

#     # Loop through each image type and save both plot and uint8 image
#     for i, (key, image) in enumerate(images_data.items()):
# #         print('image',image.shape)
#         # Normalize and convert to uint8
#         img_uint8 = (image * 255).astype(np.uint8)

#         # Save the image in its respective folder
#         image_path = os.path.join(folders[key], f"{figname}.png")
        
#         if key == "variance":  # Apply colormap to variance before saving
#             plt.imsave(image_path, image, cmap=colormaps[key])
#         else:
#             Image.fromarray(img_uint8, mode='L').save(image_path)

#         # Plot the image with the corresponding colormap
#         if key == "variance":
#             axes[i].imshow(image, cmap=colormaps[key])  # Use colormap for variance
#         else:
#             axes[i].imshow(img_uint8, cmap='gray')  # Normal grayscale images

#         axes[i].set_title(titles[key], fontsize = 18)
#         axes[i].axis('off')

#     plt.tight_layout()

#     # Save the full figure as a PDF
#     pdf_path = os.path.join(save_path, f"{figname}.pdf")
#     png_path = os.path.join(save_path, f"{figname}.png")
#     plt.savefig(png_path, dpi=300, bbox_inches='tight')
#     plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
#     # Close the plot to free memory
#     plt.close(fig)

    


def _minmax01(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def _labels_to_rgb(labels, cmap):
    """Convert integer label map (H,W) to an RGB uint8 image using a ListedColormap."""
    rgba = cmap(labels)  # (H,W,4) in [0,1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb

def plot_segmentation_results_final(
    test_images, test_annotations, out, epistemic, aleatoric, total, name, save_path, num_classes=3
):
    os.makedirs(save_path, exist_ok=True)
    figname = name[0].split('.')[0]

    folders = {
        "original": os.path.join(save_path, "original_image"),
        "ground_truth": os.path.join(save_path, "ground_truth"),
        "mean_pred": os.path.join(save_path, "mean_prediction"),
        "epistemic": os.path.join(save_path, "epistemic"),
        "aleatoric": os.path.join(save_path, "aleatoric"),
        "total": os.path.join(save_path, "total"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # ---- Original image (to HWC, uint8) ----
    orig = test_images[0].detach().cpu().numpy()           # (C,H,W) or (H,W)
    if orig.ndim == 3 and orig.shape[0] in (1, 3):         # C,H,W -> H,W,C
        orig_hwc = np.transpose(orig, (1, 2, 0))
        if orig_hwc.shape[2] == 1:
            orig_vis = (orig_hwc[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
            orig_pil = Image.fromarray(orig_vis, mode='L')
        else:
            orig_vis = (orig_hwc * 255).clip(0, 255).astype(np.uint8)
            orig_pil = Image.fromarray(orig_vis, mode='RGB')
    else:
        vis = (orig * 255).clip(0, 255).astype(np.uint8)
        if vis.ndim == 3:
            vis = vis.squeeze()
        orig_pil = Image.fromarray(vis, mode='L')

    # ---- Ground truth (binary, 0/1) ----
    gt = test_annotations[0].detach().cpu().numpy()
    if gt.ndim == 3:
        gt = np.mean(gt, axis=0)   # mean across channels
    gt = (gt > 0.5).astype(np.uint8)   # threshold to 0/1

    # ---- Mean prediction (binary, 0/1) ----
    mean_pred = out["pred_mean_argmax"]
    mean_pred = mean_pred[0] if mean_pred.ndim == 3 else mean_pred
    mean_pred = (mean_pred > 0.5).astype(np.uint8)

    # ---- Uncertainties ----
    epi = epistemic[0].detach().cpu().numpy()
    alea = aleatoric[0].detach().cpu().numpy()
    tot = total[0].detach().cpu().numpy()
    epi01 = _minmax01(epi)
    alea01 = _minmax01(alea)
    tot01 = _minmax01(tot)

    # ---- Save per-folder images ----
    orig_pil.save(os.path.join(folders["original"], f"{figname}.png"))

    # Save GT and mean_pred as black & white (0=black, 1=white)
    plt.imsave(
        os.path.join(folders["ground_truth"], f"{figname}.png"),
        gt, cmap='gray', vmin=0, vmax=1
    )
    plt.imsave(
        os.path.join(folders["mean_pred"], f"{figname}.png"),
        mean_pred, cmap='gray', vmin=0, vmax=1
    )

    # Save heatmaps
    plt.imsave(os.path.join(folders["epistemic"], f"{figname}.png"), epi01, cmap='jet')
    plt.imsave(os.path.join(folders["aleatoric"], f"{figname}.png"), alea01, cmap='jet')
    plt.imsave(os.path.join(folders["total"], f"{figname}.png"),    tot01,  cmap='jet')

    # ---- Panel figure ----
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    panels = [
        ("Original Image", np.array(orig_pil), None),
        ("Ground Truth", gt, 'gray'),
        ("Mean Prediction", mean_pred, 'gray'),
        ("Epistemic Uncertainty", epi01, 'jet'),
        ("Aleatoric Uncertainty", alea01, 'jet'),
        ("Total Uncertainty", tot01, 'jet'),
    ]
    for ax, (title, img, cmap) in zip(axes, panels):
        if cmap is None:
            if img.ndim == 3 and img.shape[2] == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap=='gray' else None)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    png_path = os.path.join(save_path, f"{figname}.png")
    pdf_path = os.path.join(save_path, f"{figname}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    
    
    
    
    
    
    
def plot_model_outputs(ref_pred, adv_preds, test_image, save_dir, filename="segmentation_outputs.png", class_index=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_image_norm = (test_image).to(device)
    
    for adv_model in range(adv_preds.shape[2]):
        plt.figure(figsize=(4 * (adv_preds.shape[1]+2), 5))
        plt.subplot(1, (adv_preds.shape[1]+2),1)
        plt.imshow(test_image[0,0].cpu().numpy(), cmap="gray")
        plt.title('Input Image')
        plt.axis("off")


        plt.subplot(1, (adv_preds.shape[1]+2),2)
        plt.imshow(torch.argmax(ref_pred, dim=1)[0].cpu().numpy(), cmap="gray")
        plt.title('reference model prediction')
        plt.axis("off")


        for i in range(adv_preds.shape[1]):
            plt.subplot(1, (adv_preds.shape[1]+2),i+ 3)
            plt.imshow(torch.argmax(adv_preds[0], dim=2)[i,adv_model].cpu().numpy(), cmap="gray")
            plt.title('Adverarial model prediction')
            plt.axis("off")

        plt.suptitle(f"QUAM Predictions")
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"model_{(adv_model+1)}_segmentation_outputs.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved model outputs to {save_path}")
        
        


def plot_mc_outputs(test_image, predictions, save_dir, filename="segmentation_outputs.png"):
    test_image_norm = (test_image).to('cuda')


    plt.figure(figsize=(4 * (len(predictions)+1), 5))
    plt.subplot(1, (len(predictions)+1),1)
    plt.imshow(test_image[0,0].cpu().numpy(), cmap="gray")
    plt.title('Input Image')
    plt.axis("off")

    for i, prediction in enumerate(predictions):
        plt.subplot(1, len(predictions)+1,i+ 2)
        plt.imshow(torch.argmax(predictions, dim=1)[i].cpu().numpy(), cmap="gray")
        plt.title(f"prediction {i+1}")
        plt.axis("off")

    plt.suptitle(f"Monte Carlo Predictions")
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved model outputs to {save_path}")
    

    
def plt_uncertainty_mc(test_image, entropy, variance, save_dir, filename="uncertainty_outputs.png"):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3,1)
    plt.imshow(test_image[0,0].cpu().numpy(), cmap="gray")
    plt.title('Input Image')
    plt.axis("off")
    plt.subplot(1, 3,2)
    plt.imshow(entropy.cpu().numpy(), cmap="hot")
    plt.title("Entropy")
    plt.colorbar()
    plt.axis("off")
    plt.subplot(1, 3,3)
    plt.imshow(variance.cpu().numpy(), cmap="hot")
    plt.title("Variance")
    plt.colorbar()
    plt.axis("off")
    
    plt.suptitle(f"Monte Carlo Uncertainty")
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved model outputs to {save_path}")

    
    
    
def plt_uncertainty(uncert,test_image, title , save_dir):
    B, C, H, W = test_image.shape
            # === Save uncertainty maps
#     print('uncert["epistemic"]',uncert["epistemic"].shape)
#     print('uncert["aleatoric"]',uncert["aleatoric"].shape)
#     print('uncert["total"]',uncert["total"].shape)
    
    epistemic_map = uncert["epistemic"].reshape(H, W).cpu().numpy()
    aleatoric_map = uncert["aleatoric"].reshape(H, W).cpu().numpy()
    total_map = uncert["total"].reshape(H, W).cpu().numpy()

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title("Test Image")
    plt.imshow(test_image[0,0], cmap='gray')
    plt.subplot(1, 4, 2)
    plt.title("Epistemic Uncertainty")
    plt.imshow(epistemic_map, cmap='hot')
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("Aleatoric Uncertainty")
    plt.imshow(aleatoric_map, cmap='hot')
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.title("Total Uncertainty")
    plt.imshow(total_map, cmap='hot')
    plt.colorbar()
    plt.suptitle(f"uncertainty maps {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"uncertainty_maps_{title}.png"))
    plt.close()

    print(f"Saved results at: {save_dir}")