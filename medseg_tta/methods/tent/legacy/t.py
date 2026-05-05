import os
import re
import glob
import gzip
import traceback
from typing import Dict, List, Tuple, Set
import numpy as np
import nibabel as nib
BASE_DIR = '/home/yuwenjing/data/imageTBAD'
IMAGE_PATTERNS = ['*_image.nii.gz', '*-image.nii.gz']
LABEL_PATTERNS = ['*_label.nii.gz', '*-label.nii.gz']
THOROUGH_READ = False
GZIP_CRC_CHECK = True

def extract_id(filename: str) -> str:
    name = os.path.basename(filename)
    for suf in ('_image.nii.gz', '-image.nii.gz', '_label.nii.gz', '-label.nii.gz'):
        if name.endswith(suf):
            return name[:-len(suf)]
    return name

def collect_ids(dir_path: str, patterns: List[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for pat in patterns:
        for fp in glob.glob(os.path.join(dir_path, pat)):
            _id = extract_id(fp)
            mapping.setdefault(_id, []).append(fp)
    return mapping

def sort_key(x: str):
    return (0, int(x)) if re.fullmatch('\\d+', x) else (1, x)

def check_gzip_crc(path: str) -> Tuple[bool, str]:
    if not path.endswith('.gz'):
        return (True, '')
    try:
        with gzip.open(path, 'rb') as f:
            while f.read(1024 * 1024):
                pass
        return (True, '')
    except Exception as e:
        return (False, f'gzip CRC failed: {type(e).__name__}: {e}')

def small_sample_slices(shape: Tuple[int, ...]) -> Tuple[slice, ...]:
    slcs = []
    for d in shape:
        mid = d // 2
        a = max(0, mid - 1)
        b = min(d, mid + 1)
        slcs.append(slice(a, b))
    return tuple(slcs)

def check_nifti_file(path: str, thorough: bool) -> Tuple[bool, List[str], dict]:
    issues: List[str] = []
    meta = {'path': path}
    if not os.path.isfile(path):
        return (False, ['file not found'], meta)
    if os.path.getsize(path) == 0:
        issues.append('empty file (size=0)')
    if GZIP_CRC_CHECK and path.endswith('.gz'):
        ok_crc, msg = check_gzip_crc(path)
        if not ok_crc:
            issues.append(msg)
    img = None
    try:
        img = nib.load(path)
    except Exception as e:
        issues.append(f'nib.load failed: {type(e).__name__}: {e}')
        return (False, issues, meta)
    try:
        meta['shape'] = tuple((int(x) for x in img.shape))
        meta['affine'] = img.affine
        meta['zooms'] = getattr(img.header, 'get_zooms', lambda: None)()
        meta['dtype'] = str(img.get_data_dtype())
    except Exception as e:
        issues.append(f'read header/meta failed: {type(e).__name__}: {e}')
    try:
        if thorough:
            data = img.get_fdata(dtype=np.float32)
        else:
            slc = small_sample_slices(img.shape)
            data = np.asanyarray(img.dataobj[slc], dtype=np.float32)
    except Exception as e:
        issues.append(f'read data failed: {type(e).__name__}: {e}')
        return (False, issues, meta)
    try:
        if not np.all(np.isfinite(data)):
            issues.append('data contains NaN/Inf')
    except Exception as e:
        issues.append(f'finite check failed: {type(e).__name__}: {e}')
    ok = len(issues) == 0
    return (ok, issues, meta)

def compare_image_label(img_path: str, lbl_path: str, thorough: bool) -> List[str]:
    problems: List[str] = []
    try:
        img = nib.load(img_path)
    except Exception as e:
        problems.append(f'image unreadable: {type(e).__name__}: {e}')
        return problems
    try:
        lbl = nib.load(lbl_path)
    except Exception as e:
        problems.append(f'label unreadable: {type(e).__name__}: {e}')
        return problems

    def normalize_shape(s):
        if len(s) == 4 and s[-1] == 1:
            return s[:3]
        return s
    s_img = normalize_shape(tuple((int(x) for x in img.shape)))
    s_lbl = normalize_shape(tuple((int(x) for x in lbl.shape)))
    if s_img != s_lbl:
        problems.append(f'shape mismatch: image {s_img} vs label {s_lbl}')
    try:
        if not np.allclose(img.affine, lbl.affine, rtol=0.0001, atol=0.001):
            problems.append('affine mismatch (image vs label)')
    except Exception as e:
        problems.append(f'affine compare failed: {type(e).__name__}: {e}')
    try:
        if thorough:
            lbl_data = lbl.get_fdata(dtype=np.float32)
        else:
            slc = small_sample_slices(lbl.shape)
            lbl_data = np.asanyarray(lbl.dataobj[slc], dtype=np.float32)
    except Exception as e:
        problems.append(f'label read failed: {type(e).__name__}: {e}')
        return problems
    if not np.all(np.isfinite(lbl_data)):
        problems.append('label contains NaN/Inf')
    if not np.allclose(lbl_data, np.round(lbl_data), atol=0.001):
        problems.append('label not integer-like (non-integer values found)')
    return problems

def main():
    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f'目录不存在: {BASE_DIR}')
    print(f'Scanning directory: {BASE_DIR}')
    image_map = collect_ids(BASE_DIR, IMAGE_PATTERNS)
    label_map = collect_ids(BASE_DIR, LABEL_PATTERNS)
    image_ids: Set[str] = set(image_map.keys())
    label_ids: Set[str] = set(label_map.keys())
    all_ids: Set[str] = image_ids | label_ids
    missing_label = sorted(list(image_ids - label_ids), key=sort_key)
    missing_image = sorted(list(label_ids - image_ids), key=sort_key)
    print(f'Found image IDs: {len(image_ids)} | label IDs: {len(label_ids)} | unique IDs: {len(all_ids)}\n')
    if missing_label:
        print('>> 有图像但缺少掩码的序号:', ', '.join(map(str, missing_label)))
    if missing_image:
        print('>> 有掩码但缺少图像的序号:', ', '.join(map(str, missing_image)))
    if not missing_label and (not missing_image):
        print('>> 图像/掩码在命名层面均成对存在。')
    dup_images = {i: files for i, files in image_map.items() if len(files) > 1}
    dup_labels = {i: files for i, files in label_map.items() if len(files) > 1}
    if dup_images or dup_labels:
        print('\n[Warning] 检测到同一序号的多份文件：')
        if dup_images:
            print('  图像重复：')
            for i in sorted(dup_images.keys(), key=sort_key):
                for p in dup_images[i]:
                    print(f'    - {os.path.basename(p)}')
        if dup_labels:
            print('  掩码重复：')
            for i in sorted(dup_labels.keys(), key=sort_key):
                for p in dup_labels[i]:
                    print(f'    - {os.path.basename(p)}')
    print('\n===== 开始逐文件损坏/可读性检查 =====')
    bad_files: List[Tuple[str, List[str]]] = []
    for _id in sorted(all_ids, key=sort_key):
        for path in image_map.get(_id, []) + label_map.get(_id, []):
            ok, issues, meta = check_nifti_file(path, thorough=THOROUGH_READ)
            if not ok:
                bad_files.append((path, issues))
                print(f'[BAD] {os.path.basename(path)} -> {issues}')
            elif issues:
                print(f'[WARN] {os.path.basename(path)} -> {issues}')
    if not bad_files:
        print('✓ 单文件读取/CRC/数值检查均通过（未发现明显损坏）。')
    else:
        print('\n!!! 发现疑似损坏/异常文件：')
        for path, issues in bad_files:
            print(f'  - {os.path.basename(path)}: {issues}')
    print('\n===== 开始图像-掩码配对一致性检查 =====')
    pair_problems = 0
    for _id in sorted(all_ids & image_ids & label_ids, key=sort_key):
        img_path = sorted(image_map[_id])[0]
        lbl_path = sorted(label_map[_id])[0]
        probs = compare_image_label(img_path, lbl_path, thorough=THOROUGH_READ)
        if probs:
            pair_problems += 1
            print(f'[PAIR-ISSUE] id={_id}')
            for p in probs:
                print(f'  - {p}')
    if pair_problems == 0:
        print('✓ 图像与掩码在形状/仿射/数值方面均一致（抽样/全量模式取决于 THOROUGH_READ）。')
    print('\n检查完成。')
    if not THOROUGH_READ:
        print('提示：如需更严格校验（完整读入 & 更强 NaN/Inf/整数性检查），将 THOROUGH_READ=True 重新运行。')
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'程序异常: {e}')
        traceback.print_exc()
