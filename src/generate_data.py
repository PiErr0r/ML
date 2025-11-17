import sys
from gen_functions import f

def main(fname):
    global f
    res = []

    for x in range(-50, 51):
        x /= 10.0
        for y in range(-50, 51):
            y /= 10.0
            res.append((x, y, round(f(x, y), 4)))

    with open(fname, 'w') as f:
        f.write('\n'.join([','.join(list(map(str, row))) for row in res]))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f"""
Usage: {sys.argv[0]} FILENAME
""")
        exit()
    main(sys.argv[1])

