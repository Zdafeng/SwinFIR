import os
import glob


def get_meta_into(test_name):
    file_names = sorted(glob.glob(os.path.join(f'{test_name}/hr/', '*')))
    name = f'meta_info_{test_name}_Test_GT.txt'
    tmp = ''
    for file_name in file_names:
        if 'hr0' in file_name:
            path = (file_name.split('/')[-1] + '\n')
            tmp += path

    with open(name, "w") as f:
        f.write(tmp)


if __name__ == '__main__':
    for test_name in ['Flickr1024', 'KITTI2012', 'KITTI2015', 'Middlebury']:
        get_meta_into(test_name)
