use encoding_rs::{GBK, UTF_16BE, UTF_16LE};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Clone, Copy, Debug)]
enum TextEncoding {
    Utf8Bom,
    Utf8,
    Utf16Le,
    Utf16Be,
    Gbk,
}

fn detect_encoding(bytes: &[u8]) -> TextEncoding {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return TextEncoding::Utf8Bom;
    }
    if bytes.starts_with(&[0xFF, 0xFE]) {
        return TextEncoding::Utf16Le;
    }
    if bytes.starts_with(&[0xFE, 0xFF]) {
        return TextEncoding::Utf16Be;
    }
    if std::str::from_utf8(bytes).is_ok() {
        TextEncoding::Utf8
    } else {
        TextEncoding::Gbk
    }
}

fn decode_text(bytes: &[u8], enc: TextEncoding) -> String {
    match enc {
        TextEncoding::Utf8Bom => String::from_utf8_lossy(&bytes[3..]).into_owned(),
        TextEncoding::Utf8 => String::from_utf8_lossy(bytes).into_owned(),
        TextEncoding::Utf16Le => {
            let (cow, _) = UTF_16LE.decode_without_bom_handling(bytes);
            cow.into_owned()
        }
        TextEncoding::Utf16Be => {
            let (cow, _) = UTF_16BE.decode_without_bom_handling(bytes);
            cow.into_owned()
        }
        TextEncoding::Gbk => {
            let (cow, _, _) = GBK.decode(bytes);
            cow.into_owned()
        }
    }
}

fn encode_text(text: &str, enc: TextEncoding) -> Vec<u8> {
    match enc {
        TextEncoding::Utf8Bom => {
            let mut out = vec![0xEF, 0xBB, 0xBF];
            out.extend_from_slice(text.as_bytes());
            out
        }
        TextEncoding::Utf8 => text.as_bytes().to_vec(),
        TextEncoding::Utf16Le => {
            let (cow, _, _) = UTF_16LE.encode(text);
            cow.into_owned()
        }
        TextEncoding::Utf16Be => {
            let (cow, _, _) = UTF_16BE.encode(text);
            cow.into_owned()
        }
        TextEncoding::Gbk => {
            let (cow, _, _) = GBK.encode(text);
            cow.into_owned()
        }
    }
}

fn normalize_ext(ext: &str) -> String {
    ext.trim()
        .trim_start_matches('.')
        .to_ascii_lowercase()
}

fn should_process(path: &Path, ext_set: &HashSet<String>) -> bool {
    let Some(ext) = path.extension().and_then(OsStr::to_str) else {
        return false;
    };
    ext_set.contains(&ext.to_ascii_lowercase())
}

fn replace_style_font_line(
    line: &str,
    from_fonts: &HashSet<String>,
    to_font: &str,
) -> (bool, String) {
    if !line.starts_with("Style:") {
        return (false, line.to_string());
    }

    // ASS/SSA Style line:
    // Style: Name,Fontname,Fontsize,...
    // Fontname is the 2nd comma-separated field (index 1).
    let mut parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 2 {
        return (false, line.to_string());
    }

    let font = parts[1].trim();
    if from_fonts.contains(font) {
        parts[1] = to_font;
        (true, parts.join(","))
    } else {
        (false, line.to_string())
    }
}

fn process_file(
    path: &Path,
    from_fonts: &HashSet<String>,
    to_font: &str,
    backup: bool,
    dry_run: bool,
) -> io::Result<bool> {
    let bytes = fs::read(path)?;
    let enc = detect_encoding(&bytes);
    let text = decode_text(&bytes, enc);

    let uses_crlf = text.contains("\r\n");
    let mut changed_any = false;
    let mut out_lines: Vec<String> = Vec::new();

    for line in text.lines() {
        let (changed, new_line) = replace_style_font_line(line, from_fonts, to_font);
        if changed {
            changed_any = true;
        }
        out_lines.push(new_line);
    }

    if !changed_any {
        return Ok(false);
    }

    let joined = if uses_crlf {
        out_lines.join("\r\n")
    } else {
        out_lines.join("\n")
    };

    if dry_run {
        return Ok(true);
    }

    if backup {
        let bak_path = path.with_extension(format!(
            "{}.bak",
            path.extension().and_then(OsStr::to_str).unwrap_or("")
        ));
        fs::write(&bak_path, &bytes)?;
    }

    let out_bytes = encode_text(&joined, enc);
    fs::write(path, out_bytes)?;
    Ok(true)
}

#[pyfunction]
fn replace_subtitle_fonts(
    from_fonts: Vec<String>,
    to_font: String,
    root_dir: String,
    file_types: Vec<String>,
    backup: Option<bool>,
    dry_run: Option<bool>,
) -> PyResult<(usize, usize)> {
    if from_fonts.is_empty() {
        return Err(PyValueError::new_err("from_fonts cannot be empty"));
    }
    if to_font.trim().is_empty() {
        return Err(PyValueError::new_err("to_font cannot be empty"));
    }
    if file_types.is_empty() {
        return Err(PyValueError::new_err("file_types cannot be empty"));
    }

    let root = PathBuf::from(&root_dir);
    if !root.exists() {
        return Err(PyValueError::new_err(format!("Directory does not exist: {}", root_dir)));
    }
    if !root.is_dir() {
        return Err(PyValueError::new_err(format!("Not a directory: {}", root_dir)));
    }

    let from_set: HashSet<String> = from_fonts.into_iter().map(|s| s.trim().to_string()).collect();
    let ext_set: HashSet<String> = file_types.into_iter().map(|e| normalize_ext(&e)).collect();

    let backup = backup.unwrap_or(true);
    let dry_run = dry_run.unwrap_or(false);

    let mut scanned: usize = 0;
    let mut modified: usize = 0;

    for entry in WalkDir::new(&root).into_iter() {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => return Err(PyOSError::new_err(format!("WalkDir error: {}", e))),
        };

        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        if !should_process(path, &ext_set) {
            continue;
        }

        scanned += 1;
        match process_file(path, &from_set, &to_font, backup, dry_run) {
            Ok(changed) => {
                if changed {
                    modified += 1;
                }
            }
            Err(e) => {
                return Err(PyOSError::new_err(format!(
                    "Failed processing {}: {}",
                    path.display(),
                    e
                )));
            }
        }
    }

    Ok((scanned, modified))
}

#[pymodule]
fn font_process(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(replace_subtitle_fonts, m)?)?;
    Ok(())
}