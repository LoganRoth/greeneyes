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
        label, height = line.split()
        height = int(height)
        h.append(height)
        tick_label.append(label)
    x = [0, 1, 2, 3, 4, 5, 6]
    w = 0.5
    plt.bar(x, h, w, tick_label=tick_label)
    plt.title('Green Eyes Analytics')
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    plt.close('all')
    return 0


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('[ERROR] Aborting')
    sys.exit(0)
