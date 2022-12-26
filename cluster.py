class Cluster:
    id = 0
    cell_list = []

    def __init__(self, id, cell_list):
        self.id = id
        self.cell_list = cell_list

    def add_cell(self, cell):
        self.cell_list.append(cell)

    def find_dist(self, other_cluster):
        s1 = set(self.cell_list)
        s2 = set(other_cluster.cell_list)

        size_s1 = len(s1)
        size_s2 = len(s2)

        intersect = s1 & s2

        size_in = len(intersect)

        jaccard_index = size_in / (size_s1 + size_s2 - size_in)

        jaccard_dist = 1 - jaccard_index

        return jaccard_dist
