#![forbid(unsafe_code)]

pub fn longest_common_prefix(strs: Vec<&str>) -> String {
    if strs.is_empty() {
        return String::new();
    }

    let first = strs[0];
    let mut prefix_end = first.len();

    for s in &strs[1..] {
        let mut common_end = 0;
        let mut chars = s.chars();

        for (i, ch1) in first.char_indices() {
            if i >= prefix_end {
                break;
            }

            match chars.next() {
                Some(ch2) if ch1 == ch2 => {
                    common_end = i + ch1.len_utf8();
                }
                _ => break,
            }
        }

        prefix_end = common_end;
        if prefix_end == 0 {
            break;
        }
    }

    first[..prefix_end].to_string()
}
