import torch
import torch.nn.functional as F


def monte_carlo_dropout_predict(model, x, num_samples):
    """
    Perform Monte Carlo dropout inference to estimate prediction variability.

    Args:
        model (nn.Module): The trained model with dropout.
        x (torch.Tensor): Input tensor (B, C, H, W).
        num_samples (int): Number of forward passes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Stacked probabilities (num_samples, B, C, H, W)
            - Stacked argmax predictions (num_samples, B, H, W)
    """
    model.train()
    predictions = []
    predictions_max = []

    with torch.no_grad():
        model.eval()
        output_ref = model(x)
        prob_ref = F.softmax(output_ref, dim=1)
        model.train()
        for _ in range(num_samples):
            
            output = model(x)
            prob = F.softmax(output, dim=1)
            pred_argmax = torch.argmax(output, dim=1)
            predictions.append(prob)
            predictions_max.append(pred_argmax)

    predictions = torch.cat(predictions, dim=0)
    predictions_max = torch.cat(predictions_max, dim=0)
    return predictions, predictions_max, prob_ref
