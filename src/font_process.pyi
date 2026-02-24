from typing import List, Tuple


def replace_subtitle_fonts(
    from_fonts: List[str],
    to_font: str,
    root_dir: str,
    file_types: List[str],
    backup: bool = True,
    dry_run: bool = False,
) -> Tuple[int, int]: ...
