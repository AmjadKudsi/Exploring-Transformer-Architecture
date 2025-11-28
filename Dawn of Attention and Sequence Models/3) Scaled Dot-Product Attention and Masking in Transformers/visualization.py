import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention_and_masks(attention_weights, masks, titles):
    """Visualize attention weights and masks"""
    fig, axes = plt.subplots(2, len(titles), figsize=(15, 6))
    
    # Plot attention weights
    for i, (attn, title) in enumerate(zip(attention_weights, titles)):
        sns.heatmap(attn[0].detach().numpy(), annot=True, fmt='.2f', 
                   cmap='Blues', ax=axes[0, i])
        axes[0, i].set_title(f'Attention: {title}')
    
    # Plot masks
    for i, (mask, title) in enumerate(zip(masks[1:], titles[1:])):
        i += 1 # skip first plot
        if mask is not None:
            if mask.dim() == 3:
                mask_viz = mask[0].float()  # Take first batch item
            elif mask.dim() == 2:
                mask_viz = mask.float()  # Already 2D
            else:
                raise ValueError(f"Unexpected mask dimension: {mask.dim()}")
            sns.heatmap(mask_viz.numpy(), annot=True, fmt='.0f', 
                       cmap='RdYlBu', ax=axes[1, i])
            axes[1, i].set_title(f'Mask: {title}')
            
    fig.delaxes(axes[1, 0])
    plt.tight_layout()
    plt.show()