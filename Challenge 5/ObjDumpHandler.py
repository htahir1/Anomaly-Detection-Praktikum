import json
import numpy as np

class ObjDumpHandler(object):
    def __init__(self, path, reset=True):
        self.path = path
        self.inverted_index = {}
        self.reset = reset

    def parse_file(self):
        opcode_master = []
        if self.reset:
            with open(self.path, 'r') as f:
                for row in f:
                    json_obj = json.loads(row)
                    try:
                        opcodes = json_obj['objdump']['sections']['.text']['blocks'][0]['opcodes']

                        sha256 = json_obj['sha256']
                        print sha256

                        if sha256 not in self.inverted_index:
                            self.inverted_index[sha256] = {}

                        for opcode in opcodes:
                            if opcode not in opcode_master:
                                opcode_master.append(opcode)

                            if opcode not in self.inverted_index[sha256]:
                                self.inverted_index[sha256][opcode] = 1
                            else:
                                self.inverted_index[sha256][opcode] += 1
                    except:
                        print "Failed"

            list_of_lists = []
            for sha256, author in sorted(self.inverted_index.items()):
                titles = []
                for opcode in opcode_master:
                    try:
                        titles.append(author[opcode])
                    except KeyError:
                        titles.append(0)
                list_of_lists.append(titles)

            data = np.array(list_of_lists)

            np.save('data/malicious_inverted_index_opcode.npy', data)
            np.savetxt("data/malicious_inverted_index_opcode.csv", np.asarray(data), delimiter=",",fmt='%.2f')
        else:
            data = np.load("data/malicious_inverted_index_opcode.npy")
            print data

