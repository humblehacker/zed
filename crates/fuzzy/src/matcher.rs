use std::{
    borrow::{Borrow, Cow},
    collections::BTreeMap,
    sync::atomic::{self, AtomicBool},
};

use crate::CharBag;

const BASE_DISTANCE_PENALTY: f64 = 0.6;
const ADDITIONAL_DISTANCE_PENALTY: f64 = 0.05;
const MIN_DISTANCE_PENALTY: f64 = 0.2;

// TODO:
// Use `Path` instead of `&str` for paths.
pub struct Matcher<'a> {
    query: &'a [char],
    lowercase_query: &'a [char],
    query_char_bag: CharBag,
    smart_case: bool,
    penalize_length: bool,
    word_boundary_boost: bool,
    min_score: f64,
    match_positions: Vec<usize>,
    last_positions: Vec<usize>,
    score_matrix: Vec<Option<f64>>,
    best_position_matrix: Vec<usize>,
}

pub trait MatchCandidate {
    fn has_chars(&self, bag: CharBag) -> bool;
    fn to_string(&self) -> Cow<'_, str>;
}

impl<'a> Matcher<'a> {
    pub fn new(
        query: &'a [char],
        lowercase_query: &'a [char],
        query_char_bag: CharBag,
        smart_case: bool,
        penalize_length: bool,
        word_boundary_boost: bool,
    ) -> Self {
        Self {
            query,
            lowercase_query,
            query_char_bag,
            min_score: 0.0,
            last_positions: vec![0; lowercase_query.len()],
            match_positions: vec![0; query.len()],
            score_matrix: Vec::new(),
            best_position_matrix: Vec::new(),
            smart_case,
            penalize_length,
            word_boundary_boost,
        }
    }

    /// Filter and score fuzzy match candidates. Results are returned unsorted, in the same order as
    /// the input candidates.
    pub(crate) fn match_candidates<C, R, F, T>(
        &mut self,
        prefix: &[char],
        lowercase_prefix: &[char],
        candidates: impl Iterator<Item = T>,
        results: &mut Vec<R>,
        cancel_flag: &AtomicBool,
        build_match: F,
    ) where
        C: MatchCandidate,
        T: Borrow<C>,
        F: Fn(&C, f64, &Vec<usize>) -> R,
    {
        let mut candidate_chars = Vec::new();
        let mut lowercase_candidate_chars = Vec::new();
        let mut extra_lowercase_chars = BTreeMap::new();

        for candidate in candidates {
            if !candidate.borrow().has_chars(self.query_char_bag) {
                continue;
            }

            if cancel_flag.load(atomic::Ordering::Relaxed) {
                break;
            }

            candidate_chars.clear();
            lowercase_candidate_chars.clear();
            extra_lowercase_chars.clear();
            for (i, c) in candidate.borrow().to_string().chars().enumerate() {
                candidate_chars.push(c);
                let mut char_lowercased = c.to_lowercase().collect::<Vec<_>>();
                if char_lowercased.len() > 1 {
                    extra_lowercase_chars.insert(i, char_lowercased.len() - 1);
                }
                lowercase_candidate_chars.append(&mut char_lowercased);
            }

            if !self.find_last_positions(lowercase_prefix, &lowercase_candidate_chars) {
                continue;
            }

            let matrix_len = self.query.len() * (prefix.len() + candidate_chars.len());
            self.score_matrix.clear();
            self.score_matrix.resize(matrix_len, None);
            self.best_position_matrix.clear();
            self.best_position_matrix.resize(matrix_len, 0);

            let score = self.score_match(
                &candidate_chars,
                &lowercase_candidate_chars,
                prefix,
                lowercase_prefix,
                &extra_lowercase_chars,
            );

            if score > 0.0 {
                results.push(build_match(
                    candidate.borrow(),
                    score,
                    &self.match_positions,
                ));
            }
        }
    }

    fn find_last_positions(
        &mut self,
        lowercase_prefix: &[char],
        lowercase_candidate: &[char],
    ) -> bool {
        let mut lowercase_prefix = lowercase_prefix.iter();
        let mut lowercase_candidate = lowercase_candidate.iter();
        for (i, char) in self.lowercase_query.iter().enumerate().rev() {
            if let Some(j) = lowercase_candidate.rposition(|c| c == char) {
                self.last_positions[i] = j + lowercase_prefix.len();
            } else if let Some(j) = lowercase_prefix.rposition(|c| c == char) {
                self.last_positions[i] = j;
            } else {
                return false;
            }
        }
        true
    }

    fn score_match(
        &mut self,
        path: &[char],
        path_lowercased: &[char],
        prefix: &[char],
        lowercase_prefix: &[char],
        extra_lowercase_chars: &BTreeMap<usize, usize>,
    ) -> f64 {
        let score = self.recursive_score_match(
            path,
            path_lowercased,
            prefix,
            lowercase_prefix,
            0,
            0,
            self.query.len() as f64,
            extra_lowercase_chars,
        ) * self.query.len() as f64;

        if score <= 0.0 {
            return 0.0;
        }
        let path_len = prefix.len() + path.len();
        let filename_start_char = path
            .iter()
            .rposition(|c| *c == std::path::MAIN_SEPARATOR)
            .map(|i| prefix.len() + i + 1)
            .unwrap_or(prefix.len());
        // Collect word boundary positions in the filename portion.
        let mut filename_boundaries: Vec<usize> = Vec::new();
        if self.word_boundary_boost {
            // Position 0 of the filename is always a boundary (start of name).
            filename_boundaries.push(filename_start_char);
            for pos in (filename_start_char + 1)..path_len {
                let prev = prefix
                    .get(pos - 1)
                    .or_else(|| path.get(pos - 1 - prefix.len()))
                    .copied();
                let curr = prefix
                    .get(pos)
                    .or_else(|| path.get(pos - prefix.len()))
                    .copied();
                if let (Some(prev), Some(curr)) = (prev, curr) {
                    let is_boundary = prev == '-'
                        || prev == '_'
                        || prev == ' '
                        || prev.is_numeric()
                        || (prev.is_lowercase() && curr.is_uppercase());
                    if is_boundary {
                        filename_boundaries.push(pos);
                    }
                }
            }
        }

        let mut cur_start = 0;
        let mut byte_ix = 0;
        let mut char_ix = 0;
        let mut filename_match_count = 0usize;
        let mut match_char_positions: Vec<usize> = Vec::new();
        for i in 0..self.query.len() {
            let match_char_ix = self.best_position_matrix[i * path_len + cur_start];
            while char_ix < match_char_ix {
                let ch = prefix
                    .get(char_ix)
                    .or_else(|| path.get(char_ix - prefix.len()))
                    .unwrap();
                byte_ix += ch.len_utf8();
                char_ix += 1;
            }

            self.match_positions[i] = byte_ix;

            if match_char_ix >= filename_start_char {
                filename_match_count += 1;
            }

            match_char_positions.push(match_char_ix);

            let matched_ch = prefix
                .get(match_char_ix)
                .or_else(|| path.get(match_char_ix - prefix.len()))
                .unwrap();
            byte_ix += matched_ch.len_utf8();

            cur_start = match_char_ix + 1;
            char_ix = match_char_ix + 1;
        }

        if self.word_boundary_boost && !self.query.is_empty() {
            let filename_ratio = filename_match_count as f64 / self.query.len() as f64;

            // Count how many query chars match the first N word boundaries
            // of the filename, where N = query length. This distinguishes
            // perfect abbreviations (SCS → S.C.S.) from scattered boundary
            // matches (SCS → S...C...S across many words).
            let mut first_boundary_hits = 0usize;
            for (qi, &match_pos) in match_char_positions.iter().enumerate() {
                if qi < filename_boundaries.len()
                    && match_pos == filename_boundaries[qi]
                {
                    first_boundary_hits += 1;
                } else {
                    break;
                }
            }
            let hump_ratio = first_boundary_hits as f64 / self.query.len() as f64;

            return score * (1.0 + filename_ratio * 0.5) * (1.0 + hump_ratio * 1.5);
        }

        score
    }

    fn recursive_score_match(
        &mut self,
        path: &[char],
        path_lowercased: &[char],
        prefix: &[char],
        lowercase_prefix: &[char],
        query_idx: usize,
        path_idx: usize,
        cur_score: f64,
        extra_lowercase_chars: &BTreeMap<usize, usize>,
    ) -> f64 {
        use std::path::MAIN_SEPARATOR;

        if query_idx == self.query.len() {
            return 1.0;
        }

        let path_len = prefix.len() + path.len();

        if let Some(memoized) = self.score_matrix[query_idx * path_len + path_idx] {
            return memoized;
        }

        let mut score = 0.0;
        let mut best_position = 0;

        let query_char = self.lowercase_query[query_idx];
        let limit = self.last_positions[query_idx];

        let max_valid_index = (prefix.len() + path_lowercased.len()).saturating_sub(1);
        let safe_limit = limit.min(max_valid_index);

        let mut last_slash = 0;
        for j in path_idx..=safe_limit {
            let extra_lowercase_chars_count = extra_lowercase_chars
                .iter()
                .take_while(|(i, _)| i < &&j)
                .map(|(_, increment)| increment)
                .sum::<usize>();
            let j_regular = j - extra_lowercase_chars_count;

            let path_char = if j < prefix.len() {
                lowercase_prefix[j]
            } else {
                let path_index = j - prefix.len();
                if path_index < path_lowercased.len() {
                    path_lowercased[path_index]
                } else {
                    continue;
                }
            };
            let is_path_sep = path_char == MAIN_SEPARATOR;

            if query_idx == 0 && is_path_sep {
                last_slash = j_regular;
            }

            #[cfg(not(target_os = "windows"))]
            let need_to_score =
                query_char == path_char || (is_path_sep && query_char == '_' || query_char == '\\');
            // `query_char == '\\'` breaks `test_match_path_entries` on Windows, `\` is only used as a path separator on Windows.
            #[cfg(target_os = "windows")]
            let need_to_score = query_char == path_char || (is_path_sep && query_char == '_');
            if need_to_score {
                let curr = if j_regular < prefix.len() {
                    prefix[j_regular]
                } else {
                    path[j_regular - prefix.len()]
                };

                let mut char_score = 1.0;
                if j > path_idx {
                    let last = if j_regular - 1 < prefix.len() {
                        prefix[j_regular - 1]
                    } else {
                        path[j_regular - 1 - prefix.len()]
                    };

                    let is_word_boundary =
                        (last == '-' || last == '_' || last == ' ' || last.is_numeric())
                            || (last.is_lowercase() && curr.is_uppercase());

                    if last == MAIN_SEPARATOR {
                        char_score = 0.9;
                    } else if is_word_boundary && self.word_boundary_boost {
                        char_score = 1.2;
                    } else if is_word_boundary {
                        char_score = 0.8;
                    } else if last == '.' {
                        char_score = 0.7;
                    } else if query_idx == 0 {
                        char_score = BASE_DISTANCE_PENALTY;
                    } else {
                        char_score = MIN_DISTANCE_PENALTY.max(
                            BASE_DISTANCE_PENALTY
                                - (j - path_idx - 1) as f64 * ADDITIONAL_DISTANCE_PENALTY,
                        );
                    }
                }

                // Apply a severe penalty if the case doesn't match.
                // This will make the exact matches have higher score than the case-insensitive and the
                // path insensitive matches.
                if (self.smart_case || curr == MAIN_SEPARATOR) && self.query[query_idx] != curr {
                    char_score *= 0.001;
                }

                let mut multiplier = char_score;

                // Scale the score based on how deep within the path we found the match.
                if self.penalize_length && query_idx == 0 {
                    multiplier /= ((prefix.len() + path.len()) - last_slash) as f64;
                }

                let mut next_score = 1.0;
                if self.min_score > 0.0 {
                    next_score = cur_score * multiplier;
                    // Scores only decrease. If we can't pass the previous best, bail
                    if next_score < self.min_score {
                        // Ensure that score is non-zero so we use it in the memo table.
                        if score == 0.0 {
                            score = 1e-18;
                        }
                        continue;
                    }
                }

                let new_score = self.recursive_score_match(
                    path,
                    path_lowercased,
                    prefix,
                    lowercase_prefix,
                    query_idx + 1,
                    j + 1,
                    next_score,
                    extra_lowercase_chars,
                ) * multiplier;

                if new_score > score {
                    score = new_score;
                    best_position = j_regular;
                    // Optimization: can't score better than 1.
                    if new_score == 1.0 {
                        break;
                    }
                }
            }
        }

        if best_position != 0 {
            self.best_position_matrix[query_idx * path_len + path_idx] = best_position;
        }

        self.score_matrix[query_idx * path_len + path_idx] = Some(score);
        score
    }
}

#[cfg(test)]
mod tests {
    use crate::{PathMatch, PathMatchCandidate};

    use super::*;
    use std::{
        path::{Path, PathBuf},
        sync::Arc,
    };

    #[test]
    fn test_get_last_positions() {
        let mut query: &[char] = &['d', 'c'];
        let mut matcher = Matcher::new(query, query, query.into(), false, true, false);
        let result = matcher.find_last_positions(&['a', 'b', 'c'], &['b', 'd', 'e', 'f']);
        assert!(!result);

        query = &['c', 'd'];
        let mut matcher = Matcher::new(query, query, query.into(), false, true, false);
        let result = matcher.find_last_positions(&['a', 'b', 'c'], &['b', 'd', 'e', 'f']);
        assert!(result);
        assert_eq!(matcher.last_positions, vec![2, 4]);

        query = &['z', '/', 'z', 'f'];
        let mut matcher = Matcher::new(query, query, query.into(), false, true, false);
        let result = matcher.find_last_positions(&['z', 'e', 'd', '/'], &['z', 'e', 'd', '/', 'f']);
        assert!(result);
        assert_eq!(matcher.last_positions, vec![0, 3, 4, 8]);
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_match_path_entries() {
        let paths = vec![
            "",
            "a",
            "ab",
            "abC",
            "abcd",
            "alphabravocharlie",
            "AlphaBravoCharlie",
            "thisisatestdir",
            "/////ThisIsATestDir",
            "/this/is/a/test/dir",
            "/test/tiatd",
        ];

        assert_eq!(
            match_single_path_query("abc", false, &paths),
            vec![
                ("abC", vec![0, 1, 2]),
                ("abcd", vec![0, 1, 2]),
                ("AlphaBravoCharlie", vec![0, 5, 10]),
                ("alphabravocharlie", vec![4, 5, 10]),
            ]
        );
        assert_eq!(
            match_single_path_query("t/i/a/t/d", false, &paths),
            vec![("/this/is/a/test/dir", vec![1, 5, 6, 8, 9, 10, 11, 15, 16]),]
        );

        assert_eq!(
            match_single_path_query("tiatd", false, &paths),
            vec![
                ("/test/tiatd", vec![6, 7, 8, 9, 10]),
                ("/this/is/a/test/dir", vec![1, 6, 9, 11, 16]),
                ("/////ThisIsATestDir", vec![5, 9, 11, 12, 16]),
                ("thisisatestdir", vec![0, 2, 6, 7, 11]),
            ]
        );
    }

    /// todo(windows)
    /// Now, on Windows, users can only use the backslash as a path separator.
    /// I do want to support both the backslash and the forward slash as path separators on Windows.
    #[cfg(target_os = "windows")]
    #[test]
    fn test_match_path_entries() {
        let paths = vec![
            "",
            "a",
            "ab",
            "abC",
            "abcd",
            "alphabravocharlie",
            "AlphaBravoCharlie",
            "thisisatestdir",
            "\\\\\\\\\\ThisIsATestDir",
            "\\this\\is\\a\\test\\dir",
            "\\test\\tiatd",
        ];

        assert_eq!(
            match_single_path_query("abc", false, &paths),
            vec![
                ("abC", vec![0, 1, 2]),
                ("abcd", vec![0, 1, 2]),
                ("AlphaBravoCharlie", vec![0, 5, 10]),
                ("alphabravocharlie", vec![4, 5, 10]),
            ]
        );
        assert_eq!(
            match_single_path_query("t\\i\\a\\t\\d", false, &paths),
            vec![(
                "\\this\\is\\a\\test\\dir",
                vec![1, 5, 6, 8, 9, 10, 11, 15, 16]
            ),]
        );

        assert_eq!(
            match_single_path_query("tiatd", false, &paths),
            vec![
                ("\\test\\tiatd", vec![6, 7, 8, 9, 10]),
                ("\\this\\is\\a\\test\\dir", vec![1, 6, 9, 11, 16]),
                ("\\\\\\\\\\ThisIsATestDir", vec![5, 9, 11, 12, 16]),
                ("thisisatestdir", vec![0, 2, 6, 7, 11]),
            ]
        );
    }

    #[test]
    fn test_lowercase_longer_than_uppercase() {
        // This character has more chars in lower-case than in upper-case.
        let paths = vec!["\u{0130}"];
        let query = "\u{0130}";
        assert_eq!(
            match_single_path_query(query, false, &paths),
            vec![("\u{0130}", vec![0])]
        );

        // Path is the lower-case version of the query
        let paths = vec!["i\u{307}"];
        let query = "\u{0130}";
        assert_eq!(
            match_single_path_query(query, false, &paths),
            vec![("i\u{307}", vec![0])]
        );
    }

    #[test]
    fn test_match_multibyte_path_entries() {
        let paths = vec![
            "aαbβ/cγdδ",
            "αβγδ/bcde",
            "c1️⃣2️⃣3️⃣/d4️⃣5️⃣6️⃣/e7️⃣8️⃣9️⃣/f",
            "/d/🆒/h",
        ];
        assert_eq!("1️⃣".len(), 7);
        assert_eq!(
            match_single_path_query("bcd", false, &paths),
            vec![
                ("αβγδ/bcde", vec![9, 10, 11]),
                ("aαbβ/cγdδ", vec![3, 7, 10]),
            ]
        );
        assert_eq!(
            match_single_path_query("cde", false, &paths),
            vec![
                ("αβγδ/bcde", vec![10, 11, 12]),
                ("c1️⃣2️⃣3️⃣/d4️⃣5️⃣6️⃣/e7️⃣8️⃣9️⃣/f", vec![0, 23, 46]),
            ]
        );
    }

    #[test]
    fn match_unicode_path_entries() {
        let mixed_unicode_paths = vec![
            "İolu/oluş",
            "İstanbul/code",
            "Athens/Şanlıurfa",
            "Çanakkale/scripts",
            "paris/Düzce_İl",
            "Berlin_Önemli_Ğündem",
            "KİTAPLIK/london/dosya",
            "tokyo/kyoto/fuji",
            "new_york/san_francisco",
        ];

        assert_eq!(
            match_single_path_query("İo/oluş", false, &mixed_unicode_paths),
            vec![("İolu/oluş", vec![0, 2, 4, 6, 8, 10, 12])]
        );

        assert_eq!(
            match_single_path_query("İst/code", false, &mixed_unicode_paths),
            vec![("İstanbul/code", vec![0, 2, 4, 6, 8, 10, 12, 14])]
        );

        assert_eq!(
            match_single_path_query("athens/şa", false, &mixed_unicode_paths),
            vec![("Athens/Şanlıurfa", vec![0, 1, 2, 3, 4, 5, 6, 7, 9])]
        );

        assert_eq!(
            match_single_path_query("BerlinÖĞ", false, &mixed_unicode_paths),
            vec![("Berlin_Önemli_Ğündem", vec![0, 1, 2, 3, 4, 5, 7, 15])]
        );

        assert_eq!(
            match_single_path_query("tokyo/fuji", false, &mixed_unicode_paths),
            vec![("tokyo/kyoto/fuji", vec![0, 1, 2, 3, 4, 5, 12, 13, 14, 15])]
        );

        let mixed_script_paths = vec![
            "résumé_Москва",
            "naïve_київ_implementation",
            "café_北京_app",
            "東京_über_driver",
            "déjà_vu_cairo",
            "seoul_piñata_game",
            "voilà_istanbul_result",
        ];

        assert_eq!(
            match_single_path_query("résmé", false, &mixed_script_paths),
            vec![("résumé_Москва", vec![0, 1, 3, 5, 6])]
        );

        assert_eq!(
            match_single_path_query("café北京", false, &mixed_script_paths),
            vec![("café_北京_app", vec![0, 1, 2, 3, 6, 9])]
        );

        assert_eq!(
            match_single_path_query("ista", false, &mixed_script_paths),
            vec![("voilà_istanbul_result", vec![7, 8, 9, 10])]
        );

        let complex_paths = vec![
            "document_📚_library",
            "project_👨‍👩‍👧‍👦_family",
            "flags_🇯🇵🇺🇸🇪🇺_world",
            "code_😀😃😄😁_happy",
            "photo_👩‍👩‍👧‍👦_album",
        ];

        assert_eq!(
            match_single_path_query("doc📚lib", false, &complex_paths),
            vec![("document_📚_library", vec![0, 1, 2, 9, 14, 15, 16])]
        );

        assert_eq!(
            match_single_path_query("codehappy", false, &complex_paths),
            vec![("code_😀😃😄😁_happy", vec![0, 1, 2, 3, 22, 23, 24, 25, 26])]
        );
    }

    #[test]
    fn test_word_boundary_boost_camel_case() {
        let paths = vec![
            "src/SimulatedChatService.rs",
            "src/SCSManager.rs",
        ];

        let results = match_single_path_query_scored("SCS", false, true, &paths);
        assert_eq!(results.len(), 2);

        let scs_manager_score = results
            .iter()
            .find(|(p, _, _)| p.contains("SCSManager"))
            .map(|(_, _, s)| *s)
            .expect("SCSManager should match");
        let simulated_score = results
            .iter()
            .find(|(p, _, _)| p.contains("SimulatedChatService"))
            .map(|(_, _, s)| *s)
            .expect("SimulatedChatService should match");

        assert!(
            simulated_score > scs_manager_score,
            "Hump match score ({simulated_score}) should beat prefix match ({scs_manager_score})"
        );
    }

    #[test]
    fn test_word_boundary_boost_hybrid_hump_prefix() {
        let paths = vec![
            "src/SimulatedChatService.rs",
            "src/SomeOtherFile.rs",
        ];

        let results = match_single_path_query_scored("SCService", false, true, &paths);
        let simulated = results
            .iter()
            .find(|(p, _, _)| p.contains("SimulatedChatService"));
        assert!(
            simulated.is_some(),
            "SCService should match SimulatedChatService"
        );
    }

    #[test]
    fn test_word_boundary_boost_snake_case() {
        let paths = vec![
            "src/simulated_chat_service.rs",
            "src/simulated-chat-service.rs",
        ];

        let results = match_single_path_query_scored("scs", false, true, &paths);
        assert_eq!(results.len(), 2, "scs should match both snake_case and kebab-case");
    }

    #[test]
    fn test_word_boundary_boost_hump_beats_prefix() {
        let paths = vec![
            "src/SimulatedChatService.rs",
            "src/SCSManager.rs",
        ];

        let results = match_single_path_query_scored("SCS", false, true, &paths);
        assert_eq!(results[0].0, "src/SimulatedChatService.rs",
            "Perfect hump match should rank first");
    }

    fn match_single_path_query_scored<'a>(
        query: &str,
        smart_case: bool,
        word_boundary_boost: bool,
        paths: &[&'a str],
    ) -> Vec<(&'a str, Vec<usize>, f64)> {
        let lowercase_query = query.to_lowercase().chars().collect::<Vec<_>>();
        let query = query.chars().collect::<Vec<_>>();
        let query_chars = CharBag::from(&lowercase_query[..]);

        let path_arcs: Vec<Arc<Path>> = paths
            .iter()
            .map(|path| Arc::from(PathBuf::from(path)))
            .collect::<Vec<_>>();
        let mut path_entries = Vec::new();
        for (i, path) in paths.iter().enumerate() {
            let lowercase_path = path.to_lowercase().chars().collect::<Vec<_>>();
            let char_bag = CharBag::from(lowercase_path.as_slice());
            path_entries.push(PathMatchCandidate {
                is_dir: false,
                char_bag,
                path: &path_arcs[i],
            });
        }

        let mut matcher =
            Matcher::new(&query, &lowercase_query, query_chars, smart_case, true, word_boundary_boost);

        let cancel_flag = AtomicBool::new(false);
        let mut results = Vec::new();

        matcher.match_candidates(
            &[],
            &[],
            path_entries.into_iter(),
            &mut results,
            &cancel_flag,
            |candidate, score, positions| PathMatch {
                score,
                worktree_id: 0,
                positions: positions.clone(),
                path: Arc::from(candidate.path),
                path_prefix: "".into(),
                distance_to_relative_ancestor: usize::MAX,
                is_dir: false,
            },
        );
        results.sort_by(|a, b| b.cmp(a));

        results
            .into_iter()
            .map(|result| {
                (
                    paths
                        .iter()
                        .copied()
                        .find(|p| result.path.as_ref() == Path::new(p))
                        .unwrap(),
                    result.positions,
                    result.score,
                )
            })
            .collect()
    }

    fn match_single_path_query<'a>(
        query: &str,
        smart_case: bool,
        paths: &[&'a str],
    ) -> Vec<(&'a str, Vec<usize>)> {
        let lowercase_query = query.to_lowercase().chars().collect::<Vec<_>>();
        let query = query.chars().collect::<Vec<_>>();
        let query_chars = CharBag::from(&lowercase_query[..]);

        let path_arcs: Vec<Arc<Path>> = paths
            .iter()
            .map(|path| Arc::from(PathBuf::from(path)))
            .collect::<Vec<_>>();
        let mut path_entries = Vec::new();
        for (i, path) in paths.iter().enumerate() {
            let lowercase_path = path.to_lowercase().chars().collect::<Vec<_>>();
            let char_bag = CharBag::from(lowercase_path.as_slice());
            path_entries.push(PathMatchCandidate {
                is_dir: false,
                char_bag,
                path: &path_arcs[i],
            });
        }

        let mut matcher = Matcher::new(&query, &lowercase_query, query_chars, smart_case, true, false);

        let cancel_flag = AtomicBool::new(false);
        let mut results = Vec::new();

        matcher.match_candidates(
            &[],
            &[],
            path_entries.into_iter(),
            &mut results,
            &cancel_flag,
            |candidate, score, positions| PathMatch {
                score,
                worktree_id: 0,
                positions: positions.clone(),
                path: Arc::from(candidate.path),
                path_prefix: "".into(),
                distance_to_relative_ancestor: usize::MAX,
                is_dir: false,
            },
        );
        results.sort_by(|a, b| b.cmp(a));

        results
            .into_iter()
            .map(|result| {
                (
                    paths
                        .iter()
                        .copied()
                        .find(|p| result.path.as_ref() == Path::new(p))
                        .unwrap(),
                    result.positions,
                )
            })
            .collect()
    }
}
