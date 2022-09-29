import re
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import pandas as pd

def my_formatter(x, pos):
    if x.is_integer():
        return str(int(x))
    else:
        return str(round(x, 5))

def add_median_labels(ax, precision='.5f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{precision}}', ha='right', va='center',
                       fontweight='bold', color='white', size=30, bbox=dict(facecolor='#445A64'))
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def print_boxplot(title="foo", which="accuracy", all_data=np.empty([30, 5]), min_y=0.9, max_y=1):
    fntDict = {'fontsize': 20, 'fontweight': 400, 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bp = ax1.boxplot([all_data[:, 0], all_data[:, 1], all_data[:, 2], all_data[:, 3], all_data[:, 4]], labels=["1", "2", "3", "4", "5"], sym='+', vert=1)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    plt.ylim([min_y, max_y])
    plt.title(title, pad=30, fontdict=fntDict)
    plt.xlabel("Number of layers")
    plt.ylabel(which + " (%)")

    medians = np.empty(5)
    for i in range(5):
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]

    box_colors = ['black', 'black']
    pos = np.arange(5) + 1
    upper_labels = [str(np.round(s, 5)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(5), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], 1.01, upper_labels[tick], transform=ax1.get_xaxis_transform(), horizontalalignment='center', size='x-small', weight=weights[k], color=box_colors[k])

    plt.show()


def print_sns_boxplot(title="foo", which="accuracy", data=np.empty([30, 5]), min_y=0.9, max_y=1):
    title = title + ' (' + which + ')'
    which = 'Accuracy (%)' if which == "Accuracy" else which
    df = pd.DataFrame(data, columns=['1', '2', '3', '4', '5', '6'])
    df = df.melt(var_name='Model', value_name=which)
    fntDict = {'fontsize': 40, 'fontweight': 400, 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}
    pd.concat([pd.concat([df], keys=['Model'], axis=1)], keys=[which])

    # sns.set_theme(style="darkgrid")
    # sns.set(rc={'figure.figsize': (18, 6)})

    sns.set_theme(style="darkgrid", palette="Spectral")
    sns.set(rc={'figure.facecolor': '#EAEAF2'}, font_scale=3)
    # sns.set_context("paper")
    # sns.set_style("darkgrid", {"axes.facecolor": "#EAEAF2", 'grid.color': 'red','xtick.direction': 'in', 'ytick.direction': 'in'})
    #print(sns.axes_style())

    # sns.violinplot(x='type', y='accuracy', palette=["m", "g"], data=data_cc)
    #which = 'Accuracy (%)' if which == "Accuracy" else which
    ax = sns.boxplot(x='Model', y=which, data=df, hue='Model')  # , width=0.3) , labels=('1', '2', '3', '4', '5', '6')
    

    ax.set_xlabel("Model")
    ax.set_ylabel(which)
    legend_label = ["RFC (original dataset)", "PCA20+RFC (original dataset)", "PCA3+RFC (original dataset)", "RFC (augmented dataset)", "PCA20+RFC (augmented dataset)", "PCA3+RFC (augmented dataset)"]
    ax.legend(title="", loc='lower left')
    print(ax.legend_.texts)
    n = 0
    for i in legend_label:
        ax.legend_.texts[n].set_text(i)
        print(n, i)
        n += 1
    sns.stripplot(x="Model", y=which, data=df, size=10, linewidth=2)

    # ax.figure.tight_layout()
    # ax.set(title=title)
    add_median_labels(ax)

    # medians = df.groupby(['Number of Layers'])[which].median()
    # vertical_offset = df[which].median() * 0.0005

    # for xtick in ax.get_xticks():
    #     ax.text(xtick, medians[xtick] + vertical_offset, round(medians[xtick], 3), horizontalalignment='center', size=12, color='w', weight='semibold')

    # axis = ax.axes
    # lines = axis.get_lines()
    # categories = axis.get_xticks()

    # for cat in categories:
    #     # every 4th line at the interval of 6 is median line
    #     # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
    #     y = round(lines[4 + cat * 6].get_ydata()[0], 4)

    #     axis.text(
    #         cat,
    #         y + 0.0005,
    #         f'{y}',
    #         ha='center',
    #         va='center',
    #         fontweight='bold',
    #         size=10,
    #         color='white',
    #         bbox=dict(facecolor='#445A64'))
    mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()
    
    plt.tight_layout()
    mng.window.state('zoomed')
    

    sns.despine()

    formatter = FuncFormatter(my_formatter)    

    ax.yaxis.grid(True)  # Hide the horizontal gridlines
    ax.yaxis.set_major_formatter(formatter)
    # ax.xaxis.grid(False)  # Show the vertical gridlines
    ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    # ax.set_title('asd')
    print(title)
    plt.title(title, pad=20, fontdict=fntDict)
    #plt.gca().legend(('y1','y2','y3','y4','y5','y6'))
    plt.ylim([min_y, max_y])
    plt.show()

if __name__ == '__main__':
    which = input('(a)ccuracy or (r)oc:\n')
    min_max_text = '0-100'
    if(which == 'r'):
        min_max_text = '0-1'

    min_y = float(input('min y (' + min_max_text + '):\n'))
    max_y = float(input('max y (' + min_max_text + '):\n'))

    pattern_acc = re.compile(r'#sa#([\S\s]*)#ea#', re.MULTILINE)
    pattern_roc = re.compile(r'#sa#([\S\s]*)#ea#', re.MULTILINE)
    all_data = np.empty([30, 6])
    counter = 0

    for file in [f for f in os.listdir('reports_rfc') if f.startswith("report_")]:
        with open(os.path.join('reports_rfc', file), 'r') as f:
            flist = f.readlines()
            #print(file)
            title = file.split("_")
            title[2] = title[2]
            print(title)
            globals()['layer%s' % title[2]] = []
            data = []
            parsing = False
            for line in flist:
                if line.startswith("#ea#" if which == "a" else "#er#"):
                    parsing = False
                if parsing:
                    data.append(float(line.strip()))
                if line.startswith("#sa#" if which == "a" else "#sr#"):
                    parsing = True
            # print(data)
            all_data[:, counter] = data
            
            counter += 1

            if(False):
                content = f.read()
                # print(content)
                # text = re.findall('(?<=#sa#)(.*?)(?=#ea#)', content, flags=re.S)
                text = re.findall(pattern_acc, content)
                # text = re.search(r'#sa#\n.*?#ea#', content)
                print(text)

    title = 'RFC and PCA+RFC' if which == "a" else 'RFC and PCA+RFC'
    if (which == "a"):
        all_data = all_data * 100
    which = "Accuracy" if which == "a" else "AUC"

    print(all_data)
    print('max:')
    print(np.max(all_data, axis = 0))  # max of each column
    print('min:')
    print(np.min(all_data, axis = 0))
    print('mean:')
    print(np.mean(all_data, axis = 0))
    print('std:')
    print(np.std(all_data, axis = 0))
    # print_boxplot(title, which, all_data)
    print_sns_boxplot(title, which, all_data, min_y, max_y)
