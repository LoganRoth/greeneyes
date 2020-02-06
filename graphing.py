import sys
import os
import matplotlib.pyplot as plt


def main():
    data = ""
    with open(os.path.join('type_array', 'array.txt'), 'r') as f:
        data = f.readlines()
    h = []
    tick_label = []
    for line in data:
        stuff = line.split()
        label = stuff[0]
        height = int(stuff[1])

        h.append(height)
        tick_label.append(label)
    x = [0, 1, 2, 3, 4, 5, 6]
    w = 0.5
    plt.bar(x, h, w, tick_label=tick_label)
    plt.title('Green Eyes Analytics')
    plt.show()
    return 0


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('[ERROR] Aborting')
    sys.exit(0)
