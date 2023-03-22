import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

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


def likert_bar_plot():
    counts_wm_t2 = [3, 5, 0, 14, 18]
    counts_mb_t2 = [6, 16, 4, 11, 3]

    counts_wm_t3 = [1, 9, 2, 14, 14]
    counts_mb_t3 = [3, 4, 1, 16, 16]

    x_labels = ['N-first WM',
                'Adaptive WM',
                'N-first MB',
                'Adaptive MB',]

    colours = ['#bebfff', '#bebfff',
               '#73b8ff', '#73b8ff']

    counts = [counts_wm_t2, counts_wm_t3, counts_mb_t2, counts_mb_t3]
    data= [[], [], [], []]

    for i in range(len(counts)):
        for j in range(len(counts[i])):
            data[i] += [j + 1] * counts[i][j]

    means = np.mean(data, axis=1)
    errors = np.std(data, axis=1)
    x_pos = np.arange(len(x_labels))
    x_pos = [0, 0.7, 2.0, 2.7]
    bar_width = [0.5 for i in range(len(counts))]

    fig, ax = plt.subplots()
    ax.bar(x_pos, means, width=bar_width, yerr=errors, align='center',
           ecolor='black', capsize=10, color=colours)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    plt.tight_layout()
    plt.show()

def likert_waffle_plot():
    df = pd.DataFrame({
        'response': ['Strongly Disagree', 'Disagree', 'Neutral',
                     'Agree', 'Strongly Agree'],
        'count': [3, 5, 0, 14, 18]
    })
    # df = pd.DataFrame({
    #     'response': ['Strongly Disagree', 'Disagree',
    #                  'Agree', 'Strongly Agree'],
    #     'count': [3, 5, 14, 18]
    # })
    total = sum(df['count'])
    proportions = [(float(value) / total) for value in df['count']]

    width = 40
    height=10
    total= width * height

    tiles_per_category = [round(proportion * total) for proportion in proportions]

    waffle = np.zeros((height, width))
    category_index = 0
    tile_index = 0
    for col in range(width):
        for row in range(height):
            tile_index += 1
            if tile_index > sum(tiles_per_category[0:category_index]):
                category_index += 1
            waffle[row, col] = category_index

    print(waffle)

    fig = plt.figure()
    colormap = plt.cm.coolwarm
    plt.matshow(waffle, cmap=colormap)
    plt.colorbar()

    fig = plt.figure()
    colormap = plt.cm.coolwarm
    plt.matshow(waffle, cmap=colormap)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, (height), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    values = df['count']
    categories = df['response']
    value_sign = ''
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')' 
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
        color_val = colormap(float(values_cumsum[i]) / total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))
        
    plt.legend(handles=legend_handles, loc = 'lower center', ncol=len(categories),
            bbox_to_anchor=(0., 0.2, 0.95, 0.1)) #positioning legends
    plt.colorbar()
    plt.show()

def likert_pywaffle_plot():
    from pywaffle import Waffle
    df = pd.DataFrame({
        'response': ['Strongly Disagree', 'Disagree', 'Neutral',
                     'Agree', 'Strongly Agree'],
        'count': [3, 5, 0, 14, 18]
    })

    counts_wm_t2 = [3, 5, 0, 14, 18]
    counts_mb_t2 = [6, 16, 4, 11, 3]

    counts_wm_t3 = [1, 9, 2, 14, 14]
    counts_mb_t3 = [3, 4, 1, 16, 16]

    counts_combined = [counts_wm_t2,
                       counts_mb_t2,
                       counts_wm_t3,
                       counts_mb_t3]

    scaled_counts = []
    for i in range(len(counts_combined)):
        proportions = [counts_combined[i][j] / 40 for j in range(len(counts_combined[i]))]
        sc = [element * 100 for element in proportions]
        print(sc)

    # Calculated scaled counts:
    s_counts_wm_t2 = [8, 12, 0, 35, 45]
    s_counts_mb_t2 = [15, 40, 10, 28, 7]
    s_counts_wm_t3 = [3, 22, 5, 35, 35]
    s_counts_mb_t3 = [8, 10, 2, 40, 40]

    s_counts_combined = [s_counts_wm_t2,
                         s_counts_mb_t2,
                         s_counts_wm_t3,
                         s_counts_mb_t3]

    colours = ['#ff8827', '#d4a35c', '#a8be91',
               '#7dd8c6', '#51f3fb']

    for i in range(len(s_counts_combined)):

        values = {
            'Strongly Disagree': s_counts_combined[i][0],
            'Disagree': s_counts_combined[i][1],
            'Neutral': s_counts_combined[i][2],
            'Agree': s_counts_combined[i][3],
            'Strongly Agree': s_counts_combined[i][4]
        }

        fig = plt.figure(
            FigureClass=Waffle, 
            rows=4, 
            values=values,
            colors=colours,
            figsize=(12, 8),
            legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)}
        )
        plt.savefig('/home/peterdu2/policy_summarization/user_study/likert_plots/'+str(i)+'.png')

if __name__ == '__main__':
    #generate_hist_task_1()
    likert_pywaffle_plot()