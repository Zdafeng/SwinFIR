from os import path as osp


def four_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """Generate four paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_L_s001.png
    0001_L_s002.png
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key_l, gt_key_l = keys
    input_key_r = input_key_l.replace('_l', '_r')
    gt_key_r = gt_key_l.replace('_l', '_r')
    
    with open(meta_info_file, 'r') as fin:
        gt_names = [line.strip().split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name.replace('_hr', '_lr'))
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(dict(
            [(f'{input_key_l}_path', input_path),
             (f'{input_key_r}_path', input_path.replace('_L', '_R').replace('_lr0', '_lr1')),
             (f'{gt_key_l}_path', gt_path),
             (f'{gt_key_r}_path', gt_path.replace('_L', '_R').replace('_hr0', '_hr1'))]))
    return paths
