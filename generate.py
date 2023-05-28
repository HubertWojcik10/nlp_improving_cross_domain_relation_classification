import os 
import numpy as np

def generate_dataset(dev=False):
    domains = ['ai', 'literature', 'news', 'politics', 'science','music']

    merged_data = []
    json_files = []
    for domain in domains:
        if dev:
            json_files.append(f'./baseline/crossre_data/{domain}-dev.json')
        else:
            json_files.append(f'./baseline/crossre_data/{domain}-train.json')

    for file in json_files:
        with open(file) as f:
            if dev:
                with open('./merged-dev.json', 'a') as f2:
                    for line in f:
                        f2.write(line)
            else:
                with open('./merged-train.json', 'a') as f2:
                    for line in f:
                        f2.write(line)

    
    print(json_files)

if __name__ == '__main__':
    generate_dataset(dev=True)