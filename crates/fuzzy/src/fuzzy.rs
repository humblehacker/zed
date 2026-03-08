mod char_bag;
mod matcher;
mod paths;
mod strings;

pub use char_bag::CharBag;
pub use paths::{
    PathMatch, PathMatchCandidate, PathMatchCandidateSet, match_fixed_path_set,
    match_fixed_path_set_with_mode, match_path_sets, match_path_sets_with_mode,
};
pub use strings::{StringMatch, StringMatchCandidate, match_strings, match_strings_with_mode};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum MatchingMode {
    #[default]
    Default,
    WordBoundaryBoosted,
}
