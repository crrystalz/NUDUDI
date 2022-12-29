# Hierarchial Clustering
def hierarchical_cluster_cells1(single_image):
    # Create a dictionary to store the coordinates of each cell, using the cell id as the key
    cell_coords = {}
    for cell in single_image.cells:
        i, j = cell.id.split("-")
        cell_coords[cell.id] = (int(i), int(j))

    # Create a list of cell histograms
    cell_histograms = []
    for cell in single_image.cells:
        cell_histograms.append(cell.histogram)

    # Calculate the distance between each pair of cells
    distances = hier.distance.pdist(cell_histograms)

    modified_distances = []

    # Generate the linkage matrix using the complete linkage method
    linkage_matrix = hier.linkage(distances, method="complete")

    # Create a list of cell ids, which will be used as labels for the dendrogram
    cell_ids = [cell.id for cell in single_image.cells]

    # Initialize the clusters dictionary, which will store the cells belonging to each cluster
    clusters = {}

    # Iterate through the rows of the linkage matrix
    for i in range(linkage_matrix.shape[0]):
        # Get the ids of the cells being merged in this row of the linkage matrix
        cell_id_1, cell_id_2 = (
            cell_ids[int(linkage_matrix[i, 0])],
            cell_ids[int(linkage_matrix[i, 1])],
        )
        
        # Get the coordinates of the cells being merged
        coord_1, coord_2 = cell_coords[cell_id_1], cell_coords[cell_id_2]
        
        # Check if the cells are adjacent
        if abs(coord_1[0] - coord_2[0]) <= 1 and abs(coord_1[1] - coord_2[1]) <= 1:
            
            # If both cells are already in different clusters, merge the clusters
            if cell_id_1 in clusters and cell_id_2 in clusters:
               
                # Get the clusters that the cells belong to
                cluster_1, cluster_2 = clusters[cell_id_1], clusters[cell_id_2]
                
                # Merge the clusters and remove the second cluster
                cluster_1.update(cluster_2)
                del clusters[cell_id_2]
            
            # If both cells are in the same cluster, do nothing
            elif cell_id_1 in clusters and cell_id_2 in clusters[cell_id_1]:
                pass
            
            # If one of the cells is already in a cluster, add the other cell to the cluster
            elif cell_id_1 in clusters:
                clusters[cell_id_1].add(cell_id_2)
                clusters[cell_id_2] = clusters[cell_id_1]
            elif cell_id_2 in clusters:
                clusters[cell_id_2].add(cell_id_1)
                clusters[cell_id_1] = clusters[cell_id_2]
            
            # If neither cell is in a cluster, create a new cluster with both cells
            else:
                clusters[cell_id_1] = [cell_id_1, cell_id_2]

            # If the cells are adjacent, include their distance in the modified distances list
            modified_distances.append(distances[i])
        else:
            # If the cells are not adjacent, don't include their distance in the modified distances list
            modified_distances.append(0)

    
    # Generate the linkage matrix using the complete linkage method
    linkage_matrix = hier.linkage(modified_distances, method="complete")

    # Plot the dendrogram
    plt.figure()
    hier.dendrogram(linkage_matrix, labels=[cell.id for cell in single_image.cells])

    # Save the dendrogram
    dir = "output\\" + single_image.filename[0:-4] + "hierarchical_clustering\\" 
    os.mkdir(dir)

    plt.savefig(dir, "dendrogram.png")

def hierarchial_cluster_cells2(single_image):
    cell_histograms = []
    for cell in single_image.cells:
        cell_histograms.append(cell.histogram)
    
    distances = hier.distance.pdist(cell_histograms)
    
    linkage_matrix = hier.linkage(distances, method="complete")
    
    plt.figure()
    hier.dendrogram(linkage_matrix, labels=[cell.id for cell in single_image.cells])
    plt.show()