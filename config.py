class Config:
    cell_size = None
    # Ratio for how many changes in cluster a cell must have to be considered changed
    min_cell_cluster_change_ratio = None
    # Minimum distance between clusters for cell to be changed
    min_cluster_dist = None

    def __init__(self, cell_size, min_cell_cluster_change_ratio, min_cluster_dist):
        self.cell_size = cell_size
        self.min_cell_cluster_change_ratio = min_cell_cluster_change_ratio
        self.min_cluster_dist = min_cluster_dist
