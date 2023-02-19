'''
    BBBEncoder (画像変換処理)

    threadingではなくconcurrent.feturesの
    GlobalInterpreterLock(GIL)の制約を受けない並列処理
'''

import re
import os
import time
import cv2
import rawpy
import flet as ft
import numpy as np
from PIL import Image
from pathlib import Path
import concurrent.futures

settings = {
    'max_workers': int(os.cpu_count()), # 並列処理数
    'jpg_comp': 96, # JPGのデフォルト圧縮率
    'png_comp': 6, # PNGのデフォルト圧縮率
    'support_input_fmt_raw': {
        'dng': '.dng',
        'nef': '.nef', # nikon
        'nrw': '.nrw', # nikon
        'crw': '.crw', # canon
        'cr2': '.cr2', # canon
        'cr3': '.cr3', # canon
        'erf': '.erf', # epson
        'orf': '.orf', # olympus
        'pef': '.pef', # pentax
        'rw2': '.rw2', # panasonic
        'arw': '.arw', # sony
        'srf': '.srf', # sony
        'sr2': '.sr2', # sony
        'dcr': '.dcr', # kodak
        'k25': '.k25', # kodak
        'kdc': '.kdc', # kodak
        'mos': '.mos', # leaf
        'mef': '.mef', # mamiya
        '3fr': '.3fr', # hasselblad
        'iiq': '.iiq', # phase one
    },
    'input_raw_opt': {
        'use_camera_wb': True,
        'half_size': False,
    },
    'support_input_fmt_pil': { # サポートされる入力フォーマット (PIL)
        'dng': 'dng',
        'png': 'dng',
        'jpg': 'jpg',
        'jpeg': 'jpeg',
        'jfif': 'jfif',
        'jp2': 'jp2',
        'tiff': 'tiff',
        'tif': 'tif',
        'webp': '.webp (WebP)',
        'bmp': 'bmp',
        'gif': 'gif',
        'dib': 'dib',
        'icns': 'icns',
        'ico': 'ico',
        'im': 'im',
        'msp': 'msp',
        'pcx': 'pcx',
        'ppm': 'ppm',
        'sgi': 'sgi',
        # 'spider': 'spider',
        'tga': 'tga',
        'xbm': 'xbm',
        'blr': 'blr',
        'cur': 'cur',
        'dcx': 'dcx',
        'fli': 'fli',
        'fpx': 'fpx',
        'frex': 'frex',
        'gbr': 'gbr',
        'gd': 'gd',
        'imt': 'imt',
        'mic': 'mic',
        'mpo': 'mpo',
        'pcd': 'pcd',
        'pixar': 'pixar',
        'psd': 'psd',
        'wal': 'wal',
        'wmf': 'wmf',
        'xpm': 'xpm',
    },
    'support_output_fmt_pil': { # サポートされる出力フォーマット (PIL)
        'bmp': '.bmp (Windows bitmaps)',
        'jpg': '.jpg (JPEG)',
        'jp2': '.jp2 (JPEG 2000)',
        'png': '.png (Portable Network Graphics)',
        'tiff': '.tiff (TIFF)',
        'webp': '.webp (WebP)',
        'gif': '.gif (Graphics Interchange Format)',
        'icns': '.icns (Apple Icon Image format)',
        'ico': '.ico',
        'im': '.im',
        # 'msp': '.msp',
        'pcx': '.pcx',
        'ppm': '.ppm',
        'sgi': '.sgi (Silicon Graphics Image)',
        # 'spider': '.spider',
        'tga': '.tga (Truevision Graphics Adapter)',
        # 'xbm': '.xbm',
        'pdf': '.pdf (Portable Document Format)',
        'dib': '.dib',
    },
    'suppport_input_fmt_cv2': { # サポートされる入力フォーマット (CV2)
        'sr': '.sr',
        'hdr': '.hdr',
        'pbm': '.pbm',
        'exr': '.exr',
    },
    'support_output_fmt_cv2': { # サポートされる出力フォーマット (CV2)
        'sr': '.sr (Sun rasters)',
        'hdr': '.hdr (Radiance HDR)',
        # 'pbm': '.pbm (Portable image format)',
        # 'exr': '.exr (OpenEXR)',
    },
}
tmp = {
    'selected_files': None,
    'encode_state': False,
}

def main(page: ft.Page):
    '''
        FletによるGUI
    '''
    page.window_width = 720
    page.window_height = 480
    page.window_resizable = False
    page.title = 'BBBEncoder'

    def pick_files_result(e: ft.FilePickerResultEvent):
        '''
            複数入力ファイル選択イベント
        '''
        selected_files = []
        for ef in e.files: selected_files.append(Path(ef.path))
        fmt_f_list, fmt_opt = get_sup_files(selected_files)
        tmp['selected_files'] = fmt_f_list
        select_i_tx.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else ""
        )
        opt = []
        for k in fmt_opt.keys():
            opt.append(
                ft.dropdown.Option(key=k, text=fmt_opt[k])
            )
        select_i_fmt.options = opt
        select_i_fmt.update()
        select_i_tx.update()
    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)

    def get_input_directory_result(e: ft.FilePickerResultEvent):
        '''
            入力ディレクトリ選択イベント
        '''
        if e.path:
            fmt_f_list, fmt_opt = get_dirs(e.path)
            tmp['selected_files'] = fmt_f_list
            opt = []
            for k in fmt_opt.keys():
                opt.append(
                    ft.dropdown.Option(key=k, text=fmt_opt[k])
                )
            select_i_fmt.options = opt
            select_i_fmt.update()
        else:
            e.path = ''
        select_i_tx.value = e.path
        select_i_tx.update()
    get_input_directory_dialog = ft.FilePicker(on_result=get_input_directory_result)

    def get_output_directory_result(e: ft.FilePickerResultEvent):
        '''
            出力ディレクトリ選択イベント
        '''
        select_o_tx.value = e.path if e.path else ""
        select_o_tx.update()
    get_output_directory_dialog = ft.FilePicker(on_result=get_output_directory_result)

    def button_clicked(e):
        '''
            エンコード実行ボタンのイベント
        '''
        if tmp['encode_state']: return
        tmp['encode_state'] = True
        start_b.text="Converting ..."
        start_b.icon=ft.icons.IMPORT_EXPORT
        start_b.update()
        input_fmt = select_i_fmt.value
        output_dir = select_o_tx.value
        output_fmt = select_o_fmt.value
        if not (input_fmt and output_dir and output_fmt): return
        if output_fmt=='jpg':
            comp_ot = settings['jpg_comp']
        elif output_fmt=='png':
            comp_ot = settings['png_comp']
        else: comp_ot = ''
        overwrite_flag = bool(overwrite_t.value)
        convert_files = tmp['selected_files'][input_fmt]
        with concurrent.futures.ProcessPoolExecutor(max_workers=settings['max_workers']) as executer:
            result = [executer.submit(
                conv,
                str(input_file), input_fmt,
                str(output_dir), output_fmt,
                comp_ot, overwrite_flag)
                for input_file in convert_files]
            for future in concurrent.futures.as_completed(result):
                pb.value += 1/len(convert_files)
                pb.update()
        # finish encode
        pb.value = 0
        pb.update()
        start_b.text="Convert"
        start_b.icon=ft.icons.PLAY_ARROW_ROUNDED
        start_b.update()
        tmp['encode_state'] = False

    def change_o_fmt(e):
        '''
            出力フォーマット変更イベント
        '''
        # print(f'{select_o_fmt} {select_o_fmt.value=="jpg"}')
        # print(comp.controls)
        if select_o_fmt.value in ['jpg', 'jp2']:
            comp_v.value = f'JPG compression: {settings["jpg_comp"]}'
            comp.controls = [
                ft.Slider(
                    min=1, max=100, divisions=99,
                    label="{value}",
                    on_change=jpg_comp_changed,
                    value=settings["jpg_comp"],
                )]
        elif select_o_fmt.value in ['png']:
            comp_v.value = f'PNG compression: {settings["png_comp"]}'
            comp.controls = [
                ft.Slider(
                    min=1, max=9, divisions=8,
                    label="{value}",
                    on_change=png_comp_changed,
                    value=settings["png_comp"],
                )]
        else:
            comp_v.value = ''
            comp.controls = [ft.Container(height=48)]
        comp_v.update()
        comp.update()

    page.overlay.extend([
        pick_files_dialog,
        get_input_directory_dialog,
        get_output_directory_dialog
    ])

    select_i_tx = ft.TextField(
        label="Input directory / files",
        hint_text="Input directory / files",
        width=580,
        read_only=True,
    )
    select_i_f = ft.IconButton(
        icon=ft.icons.IMAGE_OUTLINED,
        on_click=lambda _: pick_files_dialog.pick_files(
            allow_multiple=True
    ))
    select_i_d = ft.IconButton(
        icon=ft.icons.FOLDER_OPEN,
        on_click=lambda _: get_input_directory_dialog.get_directory_path(),
    )
    select_i_view = ft.Column(controls=[ft.Row(controls=[select_i_tx, select_i_d, select_i_f])])

    select_o_tx = ft.TextField(
        label="Output directory",
        hint_text="Output directory",
        width=580
    )
    select_o = ft.IconButton(
        icon=ft.icons.FOLDER_OPEN,
        on_click=lambda _: get_output_directory_dialog.get_directory_path(),
    )
    select_o_view = ft.Column(controls=[ft.Row(controls=[select_o_tx, select_o])])

    select_i_fmt = ft.Dropdown(
        label="Input image format",
        hint_text="Select input image format",
        width=320,
        options=[],
    )
    select_fmt_icon = ft.Icon(name=ft.icons.PLAY_ARROW_ROUNDED)
    select_o_opt = []
    for k in {**settings['support_output_fmt_pil'], **settings['support_output_fmt_cv2']}:
        select_o_opt.append(
            ft.dropdown.Option(
                key=k,
                text={
                    **settings['support_output_fmt_pil'],
                    **settings['support_output_fmt_cv2'],
                }[k]
            )
        )
    select_o_fmt = ft.Dropdown(
        label="Output image format",
        hint_text="Select output image format",
        width=320,
        on_change=change_o_fmt,
        options=select_o_opt,
    )
    select_fmt = ft.Column(controls=[ft.Row(controls=[select_i_fmt, select_fmt_icon, select_o_fmt])])

    def png_comp_changed(e):
        '''
            出力フォーマットがPNGのとき，圧縮率が変更するイベント
        '''
        settings["png_comp"] = int(e.control.value)
        comp_v.value = f'PNG compression: {int(e.control.value)}'
        comp_v.update()
    def jpg_comp_changed(e):
        '''
            出力フォーマットがJPGのとき，圧縮率が変更するイベント
        '''
        settings["jpg_comp"] = int(e.control.value)
        comp_v.value = f'JPG compression: {int(e.control.value)}'
        comp_v.update()
    comp_v = ft.Text(value="")
    comp = ft.Column([ft.Container(height=48)])

    overwrite_t = ft.Switch(label="Overwrite", value=False)
    start_b = ft.ElevatedButton(
        text="Convert",
        icon=ft.icons.PLAY_ARROW_ROUNDED,
        on_click=button_clicked,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=3),
        ),
    )
    start_view = ft.Column(controls=[ft.Row([overwrite_t, start_b], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)])
    pb = ft.ProgressBar(width=720, value=0)

    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.add(
        ft.Text(value="Image encoder settings"),
        select_i_view,
        select_o_view,
        ft.Divider(height=9, thickness=3),
        select_fmt,
        ft.Divider(height=9, thickness=3),
        comp_v,
        comp,
        start_view,
        pb,
    )

def get_dirs(host_path):
    '''
        ディレクトリ内のファイル一覧取得
    '''
    p_tmp = Path(host_path)
    ll = list(p_tmp.glob('*'))
    return get_sup_files(ll)

def get_sup_files(path_list):
    '''
        ディレクトリリスト内のサポートされている画像フォーマット一覧取得

        Parameters
        --------
        path_list : [pathlib.Path, pathlib.Path, ..]

        Returns
        --------
        fmt_f_list (separate files with format) : {'fmt': pathlib.Path, ..}
        fmt_opt (separate files with format) : {'fmt': str, ..}
    '''
    fmt_f_list, fmt_opt, sup_f_list = {}, {}, []
    for _fmt in {
        **settings['support_input_fmt_raw'],
        **settings['support_input_fmt_pil'],
        **settings['support_output_fmt_cv2'],
    }:
        gp = f'.{_fmt}$'
        ll = [p for p in path_list if re.search(gp, str(p).lower())] # 小文字で判定
        if len(ll) <= 0: continue
        fmt_f_list[_fmt] = ll
        sup_f_list.append(ll)
        tts = 's' if len(ll)>1 else ''
        fmt_opt[_fmt] = f'.{_fmt} ({len(ll)}file{tts})'
    return fmt_f_list, fmt_opt

def conv(input_file, input_fmt, output_dir, output_fmt, comp, overwrite):
    '''
        画像の保存
    '''
    fn = os.path.splitext(os.path.basename(str(input_file)))[0] # 拡張子なしのファイル名を取得
    save_p = os.path.join(output_dir, fn+f'.{output_fmt}')
    if overwrite==False and os.path.isfile(save_p): return
    try:
        if input_fmt in settings['support_input_fmt_raw']:
            with rawpy.imread(input_file) as raw:
                raw_ps = settings['input_raw_opt']
                img = raw.postprocess(
                    use_camera_wb = raw_ps['use_camera_wb'],
                    half_size = raw_ps['half_size'],
                )
        elif input_fmt in settings['support_input_fmt_pil']:
            img = Image.open(input_file)
            img = np.array(img)
        elif input_fmt in settings['support_input_fmt_cv2']:
            img = cv2.imread(input_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError('Not support format')

        os.makedirs(output_dir, exist_ok=True)
        if output_fmt in settings['support_output_fmt_pil']:
            img = Image.fromarray(img)
            if output_fmt=='jpg':
                img.save(save_p, quality=comp)
            elif output_fmt=='png':
                img.save(save_p, 'png', compress_level=int(comp))
            else:
                img.save(save_p)
        elif output_fmt in settings['support_output_fmt_cv2']:
            img = np.array(img, np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_p, img)
        else:
            raise NotImplementedError('Not support format')
    except Exception as e: print(e)
    return 0

if __name__=='__main__':
    ft.app(target=main)
