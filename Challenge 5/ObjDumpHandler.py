import json
import numpy as np

class ObjDumpHandler(object):
    def __init__(self, path, reset=False):
        self.path = path
        self.inverted_index = {}
        self.reset = reset

    def parse_file(self):
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
                            if opcode not in self.inverted_index[sha256]:
                                self.inverted_index[sha256][opcode] = 1
                            else:
                                self.inverted_index[sha256][opcode] += 1
                    except:
                        print "Failed"

            data = np.array(self.inverted_index)

            np.save('data/malicious_inverted_index_opcode.npy', data)
            np.savetxt("data/malicious_inverted_index_opcode.csv", np.asarray(data), delimiter=",",fmt='%.2f')
        else:
            data = np.load("data/malicious_inverted_index_opcode.npy")
            print data

