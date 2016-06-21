import fnmatch
import os
import random
import re
import yaml

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as colors
import matplotlib.cm as cmx

from argparse import ArgumentParser

MARKERSIZE=6

# Define arguments that can be passed at runtime
parser = ArgumentParser()
parser.add_argument('dir_path', metavar='DIR', type=str, help="path to a fitness log file(s)")
parser.add_argument('-o', '--output', type=str, default='plot', help='output file name')
parser.add_argument('-t', '--title', type=str, default='Plot Title', help='title of the plot')
parser.add_argument('-lt', '--legend-title', type=str, default='', help='the title of the legend')

parser.add_argument('--ylim-min', type=float, default=0, help='min Y axis limit')
parser.add_argument('--ylim-max', type=float, help='max Y axis limit')

parser.add_argument('--xlim-min', type=float, default=0, help='min X axis limit')
parser.add_argument('--xlim-max', type=float, help='max X axis limit')

parser.add_argument('--title-size', type=float, default=42, help='text size for the title')
parser.add_argument('--label-size', type=float, default=40, help='text size for the axis labels')
parser.add_argument('--legend-size', type=float, default=30, help='text size for the legend')
parser.add_argument('--tick-size', type=float, default=30, help='text size for the ticks')


def get_random_color():
    rand = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (rand(), rand(), rand())


def get_random_color_pretty(brightness=0.7):
    N = 30
    cmap = get_colormap(N)
    num = random.choice(range(N))
    color = cmap(num)
    # tone down a bit
    coef = brightness
    color2 = tuple([
        coef * color[0],
        coef * color[1],
        coef * color[2],
        color[3]
    ])
    return color2


def get_handles_labels(ordered_labels, color_to_label, style_to_label=None):
    legend_handles = []
    legend_labels = []
    for label in ordered_labels:
        color = color_to_label[label]

        if style_to_label is not None:
            style = style_to_label[label]
        else:
            style = ('-', '')

        hnd = mlines.Line2D([], [], color=color,
                            linestyle=style[0],
                            marker=style[1],
                            markersize=MARKERSIZE,
                            linewidth=5)

        legend_handles.append(hnd)
        legend_labels.append(label)
    return legend_handles, legend_labels


def get_label(filename):
    words = filename.split('-')
    del words[-1]
    label = ""
    for word in words:
        label += word
        label += "-"
    # chop off last dash
    return label[:-1]


def get_stylemap(N):
    styles = ['-', '--', '-.', ':']
    markers = ['', 's', '^']
    combinations = []
    for marker in markers:
        for style in styles:
            combinations.append((style, marker))

    stylemap = [combinations[i % len(combinations)] for i in range(0, N)]
    return stylemap


def get_colormap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='jet')  # hsv

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect.
        by Mark Byers: http://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main():
    args = parser.parse_args()

    title_size = args.title_size
    label_size = args.label_size
    tick_size = args.tick_size
    legend_size = args.legend_size

    dir_path = args.dir_path
    out_file_path = os.path.join(dir_path, args.output)

    files_and_dirs = os.listdir(dir_path)
    files = [f for f in files_and_dirs if os.path.isfile(os.path.join(dir_path, f)) and
             fnmatch.fnmatch(f, '*-*.log')]

    map_data_to_labels = {}
    label_set = set()

    color_to_label = {}  # dictionary {label:color} because we want the same labels have the same color
    style_to_label = {}  # dictionary {label:linestyle}

    for filename in files:
        file_path = os.path.join(dir_path, filename)
        print
        "input : {0}".format(file_path)

        with open(file_path, 'r') as in_file:
            yaml_file = in_file.read()

        # label = filename.split('-')[-2]

        label = get_label(filename)
        print
        "label : '{0}'".format(label)

        yaml_data = yaml.load(yaml_file)

        data_keyword = 'data'
        if 'velocities' in yaml_data[0]:
            data_keyword = 'velocities'
        elif 'sizes' in yaml_data[0]:
            data_keyword = 'sizes'

        data_items = [(data_item['generation'], data_item[data_keyword]) for data_item in yaml_data]
        data_items = sorted(data_items, key=lambda pair: pair[0])

        generation_num = []
        evaluation_num = []
        max_val = []
        best_val = []

        for i in range(len(data_items)):
            gen = data_items[i][0]
            data_points = data_items[i][1]
            generation_num.append(gen + 1)
            evaluation_num.append((gen + 1) * len(data_points))
            max_val.append(max(data_points) * 100.0)
            best_val.append(data_points[0])

        if label not in map_data_to_labels:
            map_data_to_labels[label] = []

        map_data_to_labels[label].append({'x': evaluation_num, 'y': max_val})
        label_set.add(label)

    sorted_labels = sorted_nicely(list(label_set))

    # assign colors to labels:
    colmap = get_colormap(len(sorted_labels))
    for i, label in enumerate(sorted_labels):
        color_to_label[label] = colmap(i)
        color_to_label[label] = 'black'

    # assign styles to labels:
    stylemap = get_stylemap(len(sorted_labels))
    for i, label in enumerate(sorted_labels):
        style_to_label[label] = stylemap[i]
        #     style_to_label[label] = ('-', '')

    # plot raw data:
    fig = plt.figure(figsize=(args.horsize, args.vertsize))
    ax = fig.add_subplot(111)

    # for label, graphs in map_data_to_labels.items():
    for label in sorted_labels:
        graphs = map_data_to_labels[label]

        for graph in graphs:
            ax.plot(graph['x'], graph['y'], linewidth=2,
                    label=label, color=color_to_label[label],
                    linestyle=style_to_label[label][0],
                    marker=style_to_label[label][1],
                    markersize=MARKERSIZE)
    hnd, lab = get_handles_labels(sorted_labels, color_to_label, style_to_label)

    lgd = ax.legend(hnd, lab, loc=0, prop={'size': legend_size}, framealpha=0.5)
    if args.legend_title is not None:
        lgd.set_title(args.legend_title, prop={'size': legend_size})

    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(args.title, fontsize=title_size, y=1.02)
    xartist = ax.set_xlabel('evaluation #', fontsize=label_size)
    yartist = ax.set_ylabel('movement speed, cm/s', fontsize=label_size)
    ax.grid()

    if args.ylim_min is not None and args.ylim_max is not None:
        ax.set_ylim(args.ylim_min, args.ylim_max)

    if args.xlim_min is not None and args.xlim_max is not None:
        ax.set_xlim(args.xlim_min, args.xlim_max)

    fig.savefig(out_file_path + ".png", bbox_extra_artists=(xartist, yartist), bbox_inches='tight')
    # ##################################################################################################

    ax = fig.add_subplot(111)

    #   for label, graphs in map_data_to_labels.items():
    for label in sorted_labels:
        graphs = map_data_to_labels[label]
        graph_lengths = [len(graph['x']) for graph in graphs]
        mean_y = []
        variance = []

        num_graphs = len(graphs)
        num_points = min(graph_lengths)

        print
        "for label '{0}'".format(label)
        print
        "{0} graphs\n{1} points".format(num_graphs, num_points)

        for i in range(num_points):
            sum = 0
            for graph in graphs:
                sum += graph['y'][i]
            sum = sum / float(num_graphs)
            mean_y.append(sum)

        # Make variance calculation
        for i in range(100, num_points, 100):
            var = 0
            for graph in graphs:
                var += (graph['y'][i] - mean_y[i]) ** 2
            var = var / float(num_graphs)
            variance.append(var)

        ax.plot(graphs[0]['x'][:num_points],
                mean_y,
                linewidth=3,
                label=label,
                color=color_to_label[label],
                linestyle=style_to_label[label][0],
                marker=style_to_label[label][1],
                markersize=MARKERSIZE)

    hnd, lab = get_handles_labels(sorted_labels, color_to_label, style_to_label)
    lgd = ax.legend(hnd, lab, loc=0, prop={'size': legend_size}, framealpha=0.5)
    if args.legend_title is not None:
        lgd.set_title(args.legend_title, prop={'size': legend_size})

    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title(args.title, fontsize=title_size, y=1.02)
    xartist = ax.set_xlabel('evaluation #', fontsize=label_size)
    yartist = ax.set_ylabel('movement speed, cm/s', fontsize=label_size)
    ax.grid()

    if args.ylim_min is not None and args.ylim_max is not None:
        ax.set_ylim(args.ylim_min, args.ylim_max)

    if args.xlim_min is not None and args.xlim_max is not None:
        ax.set_xlim(args.xlim_min, args.xlim_max)

    fig.savefig(out_file_path + "_mean.png", bbox_extra_artists=(xartist, yartist), bbox_inches='tight')


if __name__ == '__main__':
    main()
