mod char_bag;
mod matcher;
mod paths;
mod strings;

use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

pub use char_bag::CharBag;
pub use paths::{
    PathMatch, PathMatchCandidate, PathMatchCandidateSet, match_fixed_path_set, match_fixed_path_set_with_algorithm, match_path_sets, match_path_sets_with_algorithm,
};
pub use strings::{StringMatch, StringMatchCandidate, match_strings, match_strings_with_algorithm};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum FuzzyMatchingAlgorithm {
    #[default]
    Zed,
    Intellij,
}
