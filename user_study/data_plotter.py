import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def generate_hist_task_1():
    labels = [str(i) for i in range(1,10)]
    counts_n_first = [10, 10, 11, 8, 6, 9, 18, 14, 34]
    counts_adaptive = [0, 0, 2, 4, 5, 7, 14, 20, 68]
    print(counts_n_first[0]/sum(counts_n_first))
    colours = ['#ffa639', '#edb145', '#dcbc50', '#cac75c',
               '#b8d367', '#a6de73', '#95e97e', '#83f48a',
               '#71ff95']
    
    counts = counts_adaptive
    assert sum(counts) == 120

    data = []
    for i in range(len(counts)):
        data += [i for j in range(counts[i])]

    # plt.hist(data, bins=9, rwidth=0.9, align='mid')
    plt.bar(labels, counts, color=colours)
    plt.title('Adaptive Search')
    plt.ylabel('Count')
    plt.xlabel('Difficulty Rating (1 = Very Hard   9 = Very Easy)')
    plt.savefig('adaptive.png')

    plt.clf()
    counts = counts_n_first
    assert sum(counts) == 120

    data = []
    for i in range(len(counts)):
        data += [i for j in range(counts[i])]

    # plt.hist(data, bins=9, rwidth=0.9, align='mid')
    plt.bar(labels, counts, color=colours)
    plt.title('N-First Trajectories')
    plt.ylabel('Count')
    plt.xlabel('Difficulty Rating (1 = Very Hard   9 = Very Easy)')
    plt.savefig('n-first.png')

if __name__ == '__main__':
    generate_hist_task_1()