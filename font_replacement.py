import src.font_process as font_process

root = '/Volumes/Lexar/Videos/Cyber Formula'
from_fonts = [
    '微软雅黑',
    'Adobe 黑体 Std R',
    '微软雅黑 Light',
    '等线 Light',
    '等线',
    '微软正黑体',
    'PingFang SC',
    'Heiti SC',
    'simhei',
]

scanned, modified = font_process.replace_subtitle_fonts(
    from_fonts=from_fonts,
    to_font='Songti SC',
    root_dir=root,
    file_types=['ssa', 'ass'],
    backup=True,
    dry_run=False,
)

print(f'scanned={scanned}, modified={modified}')
