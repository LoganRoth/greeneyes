import sys
import os
import subprocess as sp


def main():
    type_array = [
        {'name': 'cardboard', 'count': 0},
        {'name': 'glass', 'count': 0},
        {'name': 'metal', 'count': 0},
        {'name': 'organics', 'count': 0},
        {'name': 'paper', 'count': 0},
        {'name': 'plastic', 'count': 0},
        {'name': 'trash', 'count': 0}
    ]
    grapher = sp.Popen(['python3', 'graphing.py'])
    while True:
        data = input('Enter a number: ')
        data2 = input('Enter a type: ')
        for type in type_array:
            try:
                if type['name'] == data2:
                    type['count'] += int(data)
            except Exception:
                pass
        x = ""
        for type in type_array:
            x += '{} {}\n'.format(type['name'], type['count'])
        with open(os.path.join('type_array', 'array.txt'), 'w') as f:
            f.write(x)

        # kill old script
        sp.Popen.terminate(grapher)

        # start it up again
        grapher = sp.Popen(['python3', 'graphing.py'])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('User Abort')
    sys.exit(0)
