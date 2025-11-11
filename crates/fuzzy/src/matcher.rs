use std::{
    borrow::Borrow,
    collections::BTreeMap,
    sync::atomic::{self, AtomicBool},
};

use crate::{CharBag, FuzzyMatchingAlgorithm};

#[derive(Debug, Clone, Copy)]
enum AbbreviationMatchType {
    None,
    BasicAbbreviation,
    PerfectAbbreviation,
    ExtendedAbbreviation,  // SCS + ervice
    HybridMatch,          // SC + fuzzy remainder
}

const BASE_DISTANCE_PENALTY: f64 = 0.6;
const ADDITIONAL_DISTANCE_PENALTY: f64 = 0.05;
const MIN_DISTANCE_PENALTY: f64 = 0.2;

// IntelliJ-style scoring constants
const INTELLIJ_CAMEL_CASE_BONUS: f64 = 0.95;
const INTELLIJ_WORD_BOUNDARY_BONUS: f64 = 0.85;
const INTELLIJ_ABBREVIATION_BONUS: f64 = 3.0;  // Basic abbreviation bonus
const INTELLIJ_CONSECUTIVE_BONUS: f64 = 1.1;
const INTELLIJ_PERFECT_ABBREVIATION_BONUS: f64 = 20.0;  // For consecutive camelCase abbreviations
const INTELLIJ_EXTENDED_ABBREVIATION_BONUS: f64 = 25.0;  // For abbreviation + exact remainder
const INTELLIJ_START_OF_FILENAME_BONUS: f64 = 3.0;  // Extra bonus for abbreviations starting at filename
const INTELLIJ_HYBRID_MATCH_BONUS: f64 = 8.0;  // For abbreviation + fuzzy remainder
const INTELLIJ_SCATTERED_MATCH_PENALTY: f64 = 0.1;  // Penalty for scattered non-consecutive matches

// TODO:
// Use `Path` instead of `&str` for paths.
pub struct Matcher<'a> {
    query: &'a [char],
    lowercase_query: &'a [char],
    query_char_bag: CharBag,
    smart_case: bool,
    penalize_length: bool,
    min_score: f64,
    match_positions: Vec<usize>,
    last_positions: Vec<usize>,
    score_matrix: Vec<Option<f64>>,
    best_position_matrix: Vec<usize>,
    algorithm_mode: FuzzyMatchingAlgorithm,
}

pub trait MatchCandidate {
    fn has_chars(&self, bag: CharBag) -> bool;
    fn candidate_chars(&self) -> impl Iterator<Item = char>;
}

impl<'a> Matcher<'a> {
    pub fn new(
        query: &'a [char],
        lowercase_query: &'a [char],
        query_char_bag: CharBag,
        smart_case: bool,
        penalize_length: bool,
        algorithm_mode: FuzzyMatchingAlgorithm,
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
            algorithm_mode,
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

            if cancel_flag.load(atomic::Ordering::Acquire) {
                break;
            }

            candidate_chars.clear();
            lowercase_candidate_chars.clear();
            extra_lowercase_chars.clear();
            for (i, c) in candidate.borrow().candidate_chars().enumerate() {
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
        let mut cur_start = 0;
        let mut byte_ix = 0;
        let mut char_ix = 0;
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

            let matched_ch = prefix
                .get(match_char_ix)
                .or_else(|| path.get(match_char_ix - prefix.len()))
                .unwrap();
            byte_ix += matched_ch.len_utf8();

            cur_start = match_char_ix + 1;
            char_ix = match_char_ix + 1;
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
        if query_idx == self.query.len() {
            return 1.0;
        }

        let limit = self.last_positions[query_idx];
        let max_valid_index = (prefix.len() + path_lowercased.len()).saturating_sub(1);
        let safe_limit = limit.min(max_valid_index);

        if path_idx > safe_limit {
            return 0.0;
        }

        let path_len = prefix.len() + path.len();
        if let Some(memoized) = self.score_matrix[query_idx * path_len + path_idx] {
            return memoized;
        }

        let mut score = 0.0;
        let mut best_position = 0;

        let query_char = self.lowercase_query[query_idx];

        let mut last_slash = 0;

        for j in path_idx..=safe_limit {
            let extra_lowercase_chars_count = extra_lowercase_chars
                .iter()
                .take_while(|&(&i, _)| i < j)
                .map(|(_, increment)| increment)
                .sum::<usize>();
            let j_regular = j - extra_lowercase_chars_count;

            let path_char = if j < prefix.len() {
                lowercase_prefix[j]
            } else {
                let path_index = j - prefix.len();
                match path_lowercased.get(path_index) {
                    Some(&char) => char,
                    None => continue,
                }
            };
            let is_path_sep = path_char == '/';

            if query_idx == 0 && is_path_sep {
                last_slash = j_regular;
            }
            let need_to_score = query_char == path_char || (is_path_sep && query_char == '_');
            if need_to_score {
                let curr = match prefix.get(j_regular) {
                    Some(&curr) => curr,
                    None => path[j_regular - prefix.len()],
                };

                let mut char_score = 1.0;
                if j > path_idx {
                    let last = match prefix.get(j_regular - 1) {
                        Some(&last) => last,
                        None => path[j_regular - 1 - prefix.len()],
                    };

                    match self.algorithm_mode {
                        FuzzyMatchingAlgorithm::Intellij => {
                            if last == '/' {
                                char_score = 0.9;
                            } else if last.is_lowercase() && curr.is_uppercase() {
                                // CamelCase boundary - higher priority in IntelliJ mode
                                char_score = INTELLIJ_CAMEL_CASE_BONUS;
                            } else if last == '-' || last == '_' || last == ' ' || last.is_numeric() {
                                // Word boundary
                                char_score = INTELLIJ_WORD_BOUNDARY_BONUS;
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

                            // Check for abbreviation matching in IntelliJ mode
                            let (match_type, _abbrev_length, is_at_filename_start) = Self::analyze_abbreviation_pattern(
                                &self.query,
                                &self.lowercase_query,
                                query_idx,
                                prefix,
                                path,
                                j_regular
                            );

                            match match_type {
                                AbbreviationMatchType::ExtendedAbbreviation => {
                                    char_score *= INTELLIJ_EXTENDED_ABBREVIATION_BONUS;
                                    if is_at_filename_start {
                                        char_score *= INTELLIJ_START_OF_FILENAME_BONUS;
                                    }
                                },
                                AbbreviationMatchType::PerfectAbbreviation => {
                                    char_score *= INTELLIJ_PERFECT_ABBREVIATION_BONUS;
                                    if is_at_filename_start {
                                        char_score *= INTELLIJ_START_OF_FILENAME_BONUS;
                                    }
                                },
                                AbbreviationMatchType::HybridMatch => {
                                    char_score *= INTELLIJ_HYBRID_MATCH_BONUS;
                                    if is_at_filename_start {
                                        char_score *= INTELLIJ_START_OF_FILENAME_BONUS;
                                    }
                                },
                                AbbreviationMatchType::BasicAbbreviation => {
                                    char_score *= INTELLIJ_ABBREVIATION_BONUS;
                                },
                                AbbreviationMatchType::None => {
                                    // Apply penalty for scattered matches when no abbreviation pattern found
                                    // This helps prioritize clean abbreviations over random substring matches
                                    if query_idx > 2 { // Only for longer patterns
                                        char_score *= INTELLIJ_SCATTERED_MATCH_PENALTY;
                                    }
                                }
                            }

                            // Consecutive character bonus - stronger for abbreviations
                            if query_idx > 0 && j == path_idx {
                                match match_type {
                                    AbbreviationMatchType::ExtendedAbbreviation | AbbreviationMatchType::PerfectAbbreviation => {
                                        char_score *= INTELLIJ_CONSECUTIVE_BONUS * 1.5; // Extra bonus for strong abbreviations
                                    },
                                    _ => {
                                        char_score *= INTELLIJ_CONSECUTIVE_BONUS;
                                    }
                                }
                            }
                        },
                        FuzzyMatchingAlgorithm::Zed => {
                            // Original Zed scoring logic
                            if last == '/' {
                                char_score = 0.9;
                            } else if (last == '-' || last == '_' || last == ' ' || last.is_numeric())
                                || (last.is_lowercase() && curr.is_uppercase())
                            {
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
                    }
                }

                // Apply a severe penalty if the case doesn't match.
                // This will make the exact matches have higher score than the case-insensitive and the
                // path insensitive matches.
                if (self.smart_case || curr == '/') && self.query[query_idx] != curr {
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

    fn analyze_abbreviation_pattern(
        query: &[char],
        _lowercase_query: &[char],
        query_idx: usize,
        prefix: &[char],
        path: &[char],
        current_pos: usize,
    ) -> (AbbreviationMatchType, usize, bool) {
        // Returns (match_type, abbreviation_length, is_at_filename_start)

        let get_char_at = |pos: usize| -> Option<char> {
            if pos < prefix.len() {
                prefix.get(pos).copied()
            } else {
                path.get(pos - prefix.len()).copied()
            }
        };

        // Check if we're at the start of the filename
        let is_at_filename_start = {
            let full_path_chars: Vec<char> = prefix.iter().chain(path.iter()).copied().collect();
            if let Some(last_slash_pos) = full_path_chars.iter().rposition(|&c| c == '/') {
                current_pos == last_slash_pos + 1
            } else {
                current_pos == 0
            }
        };

        // Must be at a word boundary for abbreviation
        let at_word_boundary = if current_pos == 0 {
            true
        } else {
            if let (Some(prev_char), Some(curr_char)) = (get_char_at(current_pos - 1), get_char_at(current_pos)) {
                prev_char.is_lowercase() && curr_char.is_uppercase() ||
                prev_char == '/' || prev_char == '_' || prev_char == '-' || prev_char == '.'
            } else {
                false
            }
        };

        if !at_word_boundary {
            return (AbbreviationMatchType::None, 0, false);
        }

        let remaining_query = &query[query_idx..];
        if remaining_query.is_empty() {
            return (AbbreviationMatchType::None, 0, false);
        }

        // Try to find the longest possible abbreviation match
        let mut best_abbreviation_length = 0;
        let mut best_match_type = AbbreviationMatchType::None;

        // Try different abbreviation lengths, starting from the longest possible
        for abbrev_len in (1..=remaining_query.len()).rev() {
            let abbreviation_part = &remaining_query[..abbrev_len];
            let remainder_part = &remaining_query[abbrev_len..];

            if let Some((abbreviation_end_pos, camel_matches, consecutive_score)) = Self::try_match_abbreviation(
                abbreviation_part,
                prefix,
                path,
                current_pos,
                &get_char_at
            ) {
                if camel_matches >= 2 || (camel_matches >= 1 && abbreviation_part.len() >= 2) {
                    // Found a valid abbreviation, now check what kind of match this is
                    let match_type = if remainder_part.is_empty() {
                        // Pure abbreviation (e.g., "SCS" -> "SimulatedChatService")
                        // Factor in consecutive score for better classification
                        if camel_matches >= (abbreviation_part.len() as f32 * 0.8) as usize && consecutive_score >= 4 {
                            AbbreviationMatchType::PerfectAbbreviation
                        } else {
                            AbbreviationMatchType::BasicAbbreviation
                        }
                    } else {
                        // Abbreviation + remainder (e.g., "SCService" -> "SCS" + "ervice")
                        if Self::check_exact_remainder_match(
                            remainder_part,
                            prefix,
                            path,
                            abbreviation_end_pos,
                            &get_char_at
                        ) {
                            // Factor in consecutive score for extended abbreviations too
                            if consecutive_score >= 2 {
                                AbbreviationMatchType::ExtendedAbbreviation
                            } else {
                                AbbreviationMatchType::HybridMatch
                            }
                        } else {
                            AbbreviationMatchType::HybridMatch
                        }
                    };

                    // Take the first valid match (longest abbreviation)
                    best_abbreviation_length = abbrev_len;
                    best_match_type = match_type;
                    break;
                }
            }
        }

        (best_match_type, best_abbreviation_length, is_at_filename_start)
    }

    fn try_match_abbreviation(
        abbreviation: &[char],
        prefix: &[char],
        path: &[char],
        start_pos: usize,
        get_char_at: &impl Fn(usize) -> Option<char>,
    ) -> Option<(usize, usize, usize)> {
        // Returns (end_position, camel_case_matches, consecutive_score)
        let mut query_pos = 0;
        let mut search_pos = start_pos;
        let max_search = prefix.len() + path.len();
        let mut camel_matches = 0;
        let mut consecutive_camel_score = 0;
        let mut last_match_pos = None;

        while query_pos < abbreviation.len() && search_pos < max_search {
            if let Some(search_char) = get_char_at(search_pos) {
                let query_char = abbreviation[query_pos];

                // For abbreviations, we want exact case matching for uppercase letters
                // and case-insensitive for lowercase query letters
                let is_match = if query_char.is_uppercase() {
                    // Uppercase query char must match uppercase target char exactly
                    search_char == query_char
                } else {
                    // Lowercase query char can match case-insensitively
                    search_char.to_lowercase().next() == Some(query_char.to_lowercase().next().unwrap_or('\0'))
                };

                if is_match {
                    // Check if this is a camelCase boundary (required for good abbreviations)
                    let is_camel_boundary = if search_pos == 0 {
                        search_char.is_uppercase()
                    } else if let Some(prev_char) = get_char_at(search_pos - 1) {
                        prev_char.is_lowercase() && search_char.is_uppercase()
                    } else {
                        false
                    };

                    // For uppercase query chars, we require camelCase boundaries
                    if query_char.is_uppercase() && !is_camel_boundary {
                        search_pos += 1;
                        continue;
                    }

                    if is_camel_boundary {
                        camel_matches += 1;

                        // Check for consecutive matches (higher score)
                        if let Some(last_pos) = last_match_pos {
                            if search_pos <= last_pos + 20 { // Reasonable proximity
                                consecutive_camel_score += 2;
                            }
                        }
                        last_match_pos = Some(search_pos);
                    }

                    query_pos += 1;

                    if query_pos < abbreviation.len() {
                        search_pos += 1;
                        // Look for next camelCase boundary
                        while search_pos < max_search && search_pos - start_pos < 50 {
                            if let Some(next_char) = get_char_at(search_pos) {
                                // For the next character in query
                                let next_query_char = abbreviation[query_pos];

                                if search_pos > 0 {
                                    if let Some(prev_char) = get_char_at(search_pos - 1) {
                                        let at_camel_boundary = prev_char.is_lowercase() && next_char.is_uppercase();
                                        let at_separator = prev_char == '/' || prev_char == '_' || prev_char == '-';

                                        // If next query char is uppercase, we need a camelCase boundary
                                        if next_query_char.is_uppercase() {
                                            if at_camel_boundary || at_separator || next_char == next_query_char {
                                                break;
                                            }
                                        } else {
                                            // For lowercase, be more flexible
                                            if at_camel_boundary || at_separator ||
                                               next_char.to_lowercase().next() == Some(next_query_char.to_lowercase().next().unwrap_or('\0')) {
                                                break;
                                            }
                                        }
                                    }
                                }
                                search_pos += 1;
                            } else {
                                break;
                            }
                        }
                    } else {
                        // Found all abbreviation characters
                        return Some((search_pos, camel_matches, consecutive_camel_score));
                    }
                } else {
                    search_pos += 1;
                }
            } else {
                break;
            }
        }

        if query_pos >= abbreviation.len() {
            Some((search_pos, camel_matches, consecutive_camel_score))
        } else {
            None
        }
    }

    fn check_exact_remainder_match(
        remainder: &[char],
        _prefix: &[char],
        _path: &[char],
        start_pos: usize,
        get_char_at: &impl Fn(usize) -> Option<char>,
    ) -> bool {
        // Check if remainder matches exactly at start_pos
        // For "ervice", we want to match lowercase letters in "Service"

        for offset in 0..10 { // Check a few positions after abbreviation end
            let check_pos = start_pos + offset;
            let mut matches = 0;

            for (i, &expected_char) in remainder.iter().enumerate() {
                if let Some(actual_char) = get_char_at(check_pos + i) {
                    // Case-sensitive matching for remainder
                    // If remainder char is lowercase, target should be lowercase too
                    // If remainder char is uppercase, target should be uppercase too
                    let is_match = if expected_char.is_uppercase() {
                        actual_char == expected_char
                    } else {
                        // For lowercase remainder, we want it to match the lowercase portion
                        // of the word, not the uppercase beginning
                        actual_char.is_lowercase() && actual_char == expected_char
                    };

                    if is_match {
                        matches += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            if matches == remainder.len() {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use util::rel_path::{RelPath, rel_path};

    use crate::{PathMatch, PathMatchCandidate};

    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_get_last_positions() {
        let mut query: &[char] = &['d', 'c'];
        let mut matcher = Matcher::new(query, query, query.into(), false, true, FuzzyMatchingAlgorithm::Zed);
        let result = matcher.find_last_positions(&['a', 'b', 'c'], &['b', 'd', 'e', 'f']);
        assert!(!result);

        query = &['c', 'd'];
        let mut matcher = Matcher::new(query, query, query.into(), false, true, FuzzyMatchingAlgorithm::Zed);
        let result = matcher.find_last_positions(&['a', 'b', 'c'], &['b', 'd', 'e', 'f']);
        assert!(result);
        assert_eq!(matcher.last_positions, vec![2, 4]);

        query = &['z', '/', 'z', 'f'];
        let mut matcher = Matcher::new(query, query, query.into(), false, true, FuzzyMatchingAlgorithm::Zed);
        let result = matcher.find_last_positions(&['z', 'e', 'd', '/'], &['z', 'e', 'd', '/', 'f']);
        assert!(result);
        assert_eq!(matcher.last_positions, vec![0, 3, 4, 8]);
    }

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
            "ThisIsATestDir",
            "this/is/a/test/dir",
            "test/tiatd",
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
            vec![("this/is/a/test/dir", vec![0, 4, 5, 7, 8, 9, 10, 14, 15]),]
        );

        assert_eq!(
            match_single_path_query("tiatd", false, &paths),
            vec![
                ("test/tiatd", vec![5, 6, 7, 8, 9]),
                ("ThisIsATestDir", vec![0, 4, 6, 7, 11]),
                ("this/is/a/test/dir", vec![0, 5, 8, 10, 15]),
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
            "d/🆒/h",
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

    fn match_single_path_query<'a>(
        query: &str,
        smart_case: bool,
        paths: &[&'a str],
    ) -> Vec<(&'a str, Vec<usize>)> {
        let lowercase_query = query.to_lowercase().chars().collect::<Vec<_>>();
        let query = query.chars().collect::<Vec<_>>();
        let query_chars = CharBag::from(&lowercase_query[..]);

        let path_arcs: Vec<Arc<RelPath>> = paths
            .iter()
            .map(|path| Arc::from(rel_path(path)))
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

        let mut matcher = Matcher::new(&query, &lowercase_query, query_chars, smart_case, true, FuzzyMatchingAlgorithm::Zed);

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
                path: candidate.path.into(),
                path_prefix: RelPath::empty().into(),
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
                        .find(|p| result.path.as_ref() == rel_path(p))
                        .unwrap(),
                    result.positions,
                )
            })
            .collect()
    }
}
