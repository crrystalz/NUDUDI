class OutputWriter:
    output_file = None

    def __init__(self, output_file_name):
        self.output_file = open(output_file_name, "w+")

    def close(self):
        self.output_file.close()

    def double_print(self, string):
        print(string)
        self.output_file.write(str(string) + "\n")
