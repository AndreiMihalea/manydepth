import os


og_splits_path = '/home/nemodrive/workspace/andreim/NemodriveFinal/data_split'
current_splits_path = '/home/nemodrive/workspace/andreim/manydepth/splits/upb'

og_train_scenes_path = os.path.join(og_splits_path, 'train_scenes.txt')
og_test_scenes_path = os.path.join(og_splits_path, 'test_scenes.txt')

current_train_scenes_path = os.path.join(current_splits_path, 'train_half.txt')
current_val_scenes_path = os.path.join(current_splits_path, 'val_half.txt')
current_test_scenes_path = os.path.join(current_splits_path, 'test.txt')

with open(og_train_scenes_path, 'r') as f:
    og_train_scenes = set([line.strip() for line in f.readlines()])
with open(og_test_scenes_path, 'r') as f:
    og_test_scenes = set([line.strip() for line in f.readlines()])

with open(current_train_scenes_path, 'r') as f:
    current_train_scenes = set([line.strip().split('_')[0] for line in f.readlines()])
with open(current_val_scenes_path, 'r') as f:
    current_val_scenes = set([line.strip().split('_')[0] for line in f.readlines()])
with open(current_test_scenes_path, 'r') as f:
    current_test_scenes = set([line.strip().split('_')[0] for line in f.readlines()])

assert og_train_scenes.union(og_test_scenes) == \
       current_train_scenes.union(current_test_scenes).union(current_val_scenes), \
    'Og train+test is not equal to current train+val+test'
assert og_train_scenes == current_train_scenes.union(current_val_scenes), 'Og train is not equal to current train+val'
assert og_test_scenes == current_test_scenes, 'Og test is not equal to current test'
assert current_train_scenes.intersection(current_val_scenes) == set(), 'Current train and val have common sequences'
assert current_train_scenes.intersection(current_test_scenes) == set(), 'Current train and test have common sequences'



