import argparse
import os

def fixDataset():
    parser = argparse.ArgumentParser(description='Collect root dir')
    parser.add_argument('path', metavar='path', type=str,
                        help='root dir')
    args = parser.parse_args()
    i = 0
    for subdir, dirs, files in os.walk(args.path):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            print(filepath)
            if filepath.endswith(".off"):
                with open(filepath, 'r+') as file:
                    lines = file.readlines()
                    i += 1
                    if lines[0].split("OFF")[1].split('\n')[0]:
                        line_data = lines[0].split("OFF")
                        lines[0] = "OFF\n"
                        lines.insert(1, line_data[1])
                        print('done', i)

                    file.seek(0)
                    file.writelines(lines)



if __name__ == '__main__':
    fixDataset()
