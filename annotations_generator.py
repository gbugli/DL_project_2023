import sys

def annotate(min_i,max_i,output_dir):
    with open(output_dir + 'annotations.txt', 'w') as f:
        for i in range(min_i, max_i):
            f.write(f'video_{i} 0 21 0 \n')


annotate(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])