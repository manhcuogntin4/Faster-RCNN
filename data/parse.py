import re
import os

def load_inria_annotations(index):
    filename = os.path.join('Annotations', index + '.txt')
    with open(filename) as f:
        data = f.read()
    objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', data)
    num_objs = len(objs)
    return objs

if __name__ == '__main__':
    files = ['1', '2']
    for file in files:
        print load_inria_annotations(file)
