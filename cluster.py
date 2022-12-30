class Cluster:
    id = 0
    cells = []

    def __init__(self, id):
        self.id = id
        self.cells = []

    def add_cell(self, cell):
        self.cells.append(cell)

    def has_cell(self, cell_id):
        for c in self.cells:
            if c.id == cell_id:
                return True

        return False

    def find_dist(self, other_cluster):
        s1 = set()
        s2 = set()

        for c in self.cells:
            s1.add(c.id)

        for c in other_cluster.cells:
            s2.add(c.id)

        size_s1 = len(s1)
        size_s2 = len(s2)

        intersect = s1 & s2

        size_in = len(intersect)

        jaccard_index = size_in / (size_s1 + size_s2 - size_in)

        jaccard_dist = 1 - jaccard_index

        return jaccard_dist
