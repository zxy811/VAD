#!/usr/bin/env python3
"""在目标服务器运行：先比较可执行AST，全部相符后备份并安装纯注释版本。"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

NAMES = ('VAD.py', 'VAD_head.py', 'VAD_transformer.py')
HERE = Path(__file__).resolve().parent

def syntax_tree(text, filename):
    return ast.dump(ast.parse(text, filename=filename), include_attributes=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default='/root/autodl-tmp/VAD/projects/mmdet3d_plugin/VAD')
    parser.add_argument('--check-only', action='store_true', help='只检查，不写入')
    args = parser.parse_args()
    target = Path(args.target).resolve()
    prepared = {}
    errors = []
    for name in NAMES:
        dst = target / name
        if not dst.is_file():
            errors.append('目标文件不存在：' + str(dst))
            continue
        current = dst.read_text(encoding='utf-8')
        original = (HERE / 'originals' / name).read_text(encoding='utf-8')
        annotated = (HERE / 'annotated' / name).read_text(encoding='utf-8')
        expected = syntax_tree(original, name)
        if syntax_tree(annotated, name) != expected:
            errors.append('交付包内部AST校验失败：' + name)
        elif syntax_tree(current, name) != expected:
            errors.append('服务器文件的可执行代码与基线不同，已保护不覆盖：' + name)
        else:
            compile(annotated, name, 'exec')
            prepared[name] = annotated
            print('检查通过：', dst)
    if errors:
        print('\n'.join(errors))
        raise SystemExit('没有替换任何文件。请提供当前三份源码，以保留调试打印或其他修改后重新注释。')
    if args.check_only:
        print('三份文件全部相符。check-only模式未写入任何内容。')
        return
    backup = target / ('annotation_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir()
    for name in NAMES:
        shutil.copy2(str(target / name), str(backup / name))
    replaced = []
    try:
        for name in NAMES:
            dst = target / name
            mode = dst.stat().st_mode
            fd, tmp_name = tempfile.mkstemp(prefix='.' + name + '.', dir=str(target))
            try:
                with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
                    stream.write(prepared[name])
                os.chmod(tmp_name, mode)
                os.replace(tmp_name, str(dst))
                replaced.append(name)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        for name in NAMES:
            current = (target / name).read_text(encoding='utf-8')
            assert syntax_tree(current, name) == syntax_tree(prepared[name], name)
    except BaseException:
        for name in replaced:
            shutil.copy2(str(backup / name), str(target / name))
        raise
    print('安装完成；原文件备份：', backup)
    print('仅注释变化，AST一致；未改动配置、依赖、权重，也没有启动训练。')
    print('回滚命令（请先停止相关Python进程再执行）：')
    for name in NAMES:
        print('cp ' + str(backup / name) + ' ' + str(target / name))

if __name__ == '__main__':
    main()
