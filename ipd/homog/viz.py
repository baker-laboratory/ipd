import numpy as np
import ipd

@ipd.dev.iterize_on_first_param(basetype=np.ndarray)
def can_showme(arg, **kwargs):
    if not isinstance(arg, np.ndarray): return False
    if ipd.homog.is_contact_matrix(arg): return True
    return False

@ipd.dev.iterize_on_first_param(basetype=np.ndarray)
def showme(arg, **kwargs):
    if ipd.homog.is_contact_matrix(arg):
        return visualize_contact_maps(arg, **kwargs)
    raise ValueError('Invalid input. Expected a stack of contact maps.')

def visualize_contact_maps(contact_maps, n_cols=3, figsize=(15, 15), cmap='viridis'):
    """
    Visualize a stack of contact maps using seaborn heatmap.

    Parameters:
        contact_maps (np.ndarray): 3D numpy array of shape (m, n, n) where m is the number of contact maps.
        n_cols (int): Number of columns in the subplot grid.
        figsize (tuple): Size of the entire figure.
        cmap (str): Colormap to use for the heatmaps.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    m = contact_maps.shape[0]
    n_rows = math.ceil(m / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # If there's only one row/column, ensure axes is iterable
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i in range(m):
        ax = axes[i]
        sns.heatmap(contact_maps[i], ax=ax, cmap=cmap, square=True, cbar=True)
        ax.set_title(f'Contact Map {i+1}')
        # Optionally, remove ticks for clarity
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove any unused subplots
    for j in range(m, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
