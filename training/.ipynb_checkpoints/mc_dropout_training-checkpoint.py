import torch


def monte_carlo_dropout_predict(model, image, num_samples: int = 10):
    """
    Perform Monte Carlo Dropout inference.

    Args:
        model: segmentation model with dropout layers
        image: input tensor [B,C,H,W]
        num_samples: number of stochastic forward passes

    Returns:
        predictions: stacked predictions [num_samples, B, C, H, W]
        predictions_max: argmax segmentation masks [num_samples, B, H, W]
        prob_ref: averaged probabilities [B, C, H, W]
    """
    model.train()  # dropout ON during inference

    preds = []
    for _ in range(num_samples):
        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            preds.append(probs)

    predictions = torch.stack(preds, dim=0)  # [num_samples, B, C, H, W]
    predictions_max = torch.argmax(predictions, dim=2)  # [num_samples, B, H, W]
    prob_ref = predictions.mean(dim=0)  # averaged probability

    return predictions, predictions_max, prob_ref
