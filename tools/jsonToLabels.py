import json
import os

with open("./dataset/ShapeNetCore.v2/ShapeNetCore.v2/taxonomy.json","r") as f:
    raw_json = json.load(f)

# print(raw_json[0]['synsetId']))
# f.close()

rootdir = './dataset/ShapeNetCore.v2/ShapeNetCore.v2'

#print(os.listdir(rootdir))

for dir in os.listdir(rootdir):
    for i in range(len(raw_json)):
        if dir == raw_json[i]['synsetId']:
            name = raw_json[i]['name'].split(',')
            old_path = os.path.join(rootdir,dir)
            new_path = os.path.join(rootdir,name[0])
            os.rename(old_path,new_path)


# for subdir, dirs, files in os.walk(rootdir):
#     print(subdir)
