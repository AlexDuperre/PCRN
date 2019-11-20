# A simple script that copy files to an new root with the same structure as the initial root.
#
# Example:
# find . -name *.png -print0 | xargs -0 -n1 -P3 -I {} python Dataset_FileMover.py -- {}
#

import argparse, sys, os
from shutil import copyfile

parser = argparse.ArgumentParser(description='Copy png depth files to other directory with same structure.')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')

args = parser.parse_args()
base_output_folder = "../ShapeNetCoreV2 - Depth"

output_dir = os.path.join(base_output_folder,*args.obj.split('/')[1:-1])
output_file = os.path.join(base_output_folder,*args.obj.split('/')[1:])


if ("image" not in output_dir) and ("screenshot" not in output_dir):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        pass

    copyfile(args.obj,output_file)




