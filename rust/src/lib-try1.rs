use pyo3::prelude::*;

use std::collections::{BTreeMap, BTreeSet};
use std::cmp::Ordering;
use std::sync::Arc;

// Define a type alias for the score vector
type ScoreVector = (bool, f64, f64, i64);

// Define a struct for RuleInfo
#[pyclass]
struct RuleInfo {
    min_matched_length_unique: i64,
    min_matched_length: i64,
    min_high_matched_length_unique: i64,
    min_high_matched_length: i64,
    minimum_containment: f64,
}


// pub trait SCSet {
//     fn intersect(&self, other: &Self) -> Self;
//     fn counter(&self) -> usize;
// }

// #[pyclass]
// struct BitSet {
//     set: Arc<BTreeSet<i64>>
// }

// #[pymethods]
// impl SCSet for BitSet {
//     fn intersect(&self, other: &Self) -> BitSet {
//         BitSet { set: Arc::new(self.set.intersection(&other.set).cloned().collect()) }
//     }

//     fn counter(&self) -> usize {
//         self.set.len()
//     }
// }

// #[pyclass]
// struct MSet {
//     set: Arc<BTreeMap<i64, i64>>
// }

// #[pymethods]
// impl BitSet {
//     fn intersect(&self, other: &Self) -> BitSet {
//         BitSet { set: Arc::new(self.set.intersection(&other.set).cloned().collect()) }
//     }

//     fn counter(&self) -> usize {
//         self.set.len()
//     }
// }


// Implement the build_set_and_tids_mset function
#[pyfunction]
fn build_set_and_tids_mset(token_ids: Vec<i64>) -> (BTreeSet<i64>, BTreeMap<i64, i64>) {
    let mut tids_mset = BTreeMap::new();
    let mut tids_set = BTreeSet::new();

    for tid in token_ids {
        if tid != -1 {
            *tids_mset.entry(tid).or_insert(0) += 1;
            tids_set.insert(tid);
        }
    }

    (tids_set, tids_mset)
}   

// Implement the set_counter function
fn set_counter(set: &BTreeSet<i64>) -> usize {
    set.len()
}

fn set_counter_mset(set: &BTreeMap<i64, i64>) -> i64 {
    set.values().sum::<i64>()
}

// Implement the set_intersector function
fn set_intersector(set1: &BTreeSet<i64>, set2: &BTreeSet<i64>) -> BTreeSet<i64> {
    set1.intersection(set2).cloned().collect()
}

fn set_intersector_mset(set1: &BTreeMap<i64, i64>, set2: &BTreeMap<i64, i64>) -> BTreeMap<i64, i64> {
    let mut ret = BTreeMap::new();
    for (k, v) in set1 {
        if let Some(v2) = set2.get(k) {
            ret.insert(*k, std::cmp::min(*v, *v2));
        }
    }
    ret
}

// Implement the set_high_intersection_filter function
fn set_high_intersection_filter(set: &BTreeSet<i64>, cutoff: i64) -> BTreeSet<i64> {
    set.iter().filter(|x| **x <= cutoff).cloned().collect()
}

fn set_high_intersection_filter_mset(set: &BTreeMap<i64, i64>, cutoff: i64) -> BTreeMap<i64, i64> {
    set.iter().filter(|(k, _)| **k <= cutoff).map(|(k, v)| (*k, *v)).collect()
}

// Implement the compare_token_sets function
fn compare_token_sets(
    qset: &BTreeSet<i64>,
    iset: &BTreeSet<i64>,
    len_legalese: i64,
    min_matched_length_high: i64,
    min_matched_length: i64,
    minimum_containment: f64,
    high_resemblance_threshold: f64,
) -> Option<(ScoreVector, ScoreVector, BTreeSet<i64>)> {
    let intersection = set_intersector(qset, iset);
    if intersection.is_empty() {
        return None;
    }

    let high_intersection = set_high_intersection_filter(&intersection, len_legalese);
    if high_intersection.is_empty() {
        return None;
    }

    if (set_counter(&high_intersection) as i64) < min_matched_length_high {
        return None;
    }

    let rule_length = set_counter(iset);
    let matched_length = set_counter(&intersection);
    if (matched_length as i64) < min_matched_length {
        return None;
    }

    let union_len = set_counter(qset) + rule_length - matched_length;
    let resemblance = matched_length as f64 / union_len as f64;
    let containment = matched_length as f64 / rule_length as f64;
    if containment < minimum_containment {
        return None;
    }

    let amplified_resemblance = resemblance.powi(2);
    let score_vec1 = (
        (resemblance * 10.0).round() as i64 / 10 == (high_resemblance_threshold * 10.0).round() as i64 / 10,
        (containment * 10.0).round() / 10.0,
        (amplified_resemblance * 10.0).round() / 10.0,
        (matched_length as f64 / 20.0).round() as i64,
    );

    let score_vec2 = (
        resemblance >= high_resemblance_threshold,
        containment,
        amplified_resemblance,
        matched_length as i64,
    );

    Some((score_vec1, score_vec2, high_intersection))
}

// Implement the compare_token_sets_mset function
fn compare_token_sets_mset(
    qset: &BTreeMap<i64, i64>,
    iset: &BTreeMap<i64, i64>,
    len_legalese: i64,
    min_matched_length_high: i64,
    min_matched_length: i64,
    minimum_containment: f64,
    high_resemblance_threshold: f64,
) -> Option<(ScoreVector, ScoreVector, BTreeMap<i64, i64>)> {
    let intersection = set_intersector_mset(qset, iset);
    if intersection.is_empty() {
        return None;
    }

    let high_intersection = set_high_intersection_filter_mset(&intersection, len_legalese);
    if high_intersection.is_empty() {
        return None;
    }

    if set_counter_mset(&high_intersection) < min_matched_length_high {
        return None;
    }

    let rule_length = set_counter_mset(iset);
    let matched_length = set_counter_mset(&intersection);
    if matched_length < min_matched_length {
        return None;
    }

    let union_len = set_counter_mset(qset) + rule_length - matched_length;
    let resemblance = matched_length as f64 / union_len as f64;
    let containment = matched_length as f64 / rule_length as f64;
    if containment < minimum_containment {
        return None;
    }

    let amplified_resemblance = resemblance.powi(2);
    let score_vec1 = (
        (resemblance * 10.0).round() as i64 / 10 == (high_resemblance_threshold * 10.0).round() as i64 / 10,
        (containment * 10.0).round() / 10.0,
        (amplified_resemblance * 10.0).round() / 10.0,
        (matched_length as f64 / 20.0).round() as i64,
    );

    let score_vec2 = (
        resemblance >= high_resemblance_threshold,
        containment,
        amplified_resemblance,
        matched_length as i64,
    );

    Some((score_vec1, score_vec2, high_intersection))
}

// Implement the compute_candidates function
#[pyfunction]
fn compute_candidates(
    token_ids: Vec<i64>,
    len_legalese: i64,
    rules_by_rid: Vec<RuleInfo>,
    sets_by_rid: Vec<Option<BTreeSet<i64>>>,
    msets_by_rid: Vec<Option<BTreeMap<i64, i64>>>,
    matchable_rids: BTreeSet<i64>,
    top: usize,
    high_resemblance: bool,
    high_resemblance_threshold: f64,
) -> Vec<(ScoreVector, ScoreVector, i64, RuleInfo, BTreeSet<i64>)> {
    let (qset, qmset) = build_set_and_tids_mset(token_ids);

    let mut sortable_candidates: Vec<(ScoreVector, ScoreVector, i64, RuleInfo, BTreeSet<i64>)> = Vec::new();

    for (rid, rule) in rules_by_rid.iter().enumerate() {
        let rid = rid as i64;
        if !matchable_rids.contains(&rid) {
            continue;
        }

        let set = sets_by_rid.get(rid as usize).unwrap();
        if set.is_none() {
            continue;
        }
        let set = set.as_ref().unwrap();

        let scores_vectors = compare_token_sets(
            &qset,
            set,
            len_legalese,
            rule.min_high_matched_length_unique,
            rule.min_matched_length_unique,
            rule.minimum_containment,
            high_resemblance_threshold,
        );

        if let Some((score_vec1, score_vec2, high_set_intersection)) = scores_vectors {
            if !high_resemblance || (high_resemblance && score_vec1.0 && score_vec2.0) {
                sortable_candidates.push((score_vec1, score_vec2, rid, rule.clone(), high_set_intersection));
            }
        }
    }

    if sortable_candidates.is_empty() {
        return sortable_candidates;
    }

    sortable_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut sortable_candidates_new: Vec<(ScoreVector, ScoreVector, i64, RuleInfo, BTreeSet<i64>)> = Vec::new();

    for (k, (_,_, rid, rule, high_set_intersection)) in sortable_candidates.iter().enumerate() {
        if k >= 10 * top {
            break;
        }

        let mset = msets_by_rid.get(*rid as usize).unwrap();
        if mset.is_none() {
            continue;
        }
        let mset = mset.as_ref().unwrap();

        let scores_vectors = compare_token_sets_mset(
            &qmset,
            mset,
            len_legalese,
            rule.min_high_matched_length,
            rule.min_matched_length,
            rule.minimum_containment,
            high_resemblance_threshold,
        );

        if let Some((score_vec1, score_vec2, _intersection)) = scores_vectors {
            if !high_resemblance || (high_resemblance && score_vec1.0 && score_vec2.0) {
                sortable_candidates_new.push((score_vec1, score_vec2, *rid, rule.clone(), high_set_intersection.clone()));
            }
        }
    }

    if sortable_candidates_new.is_empty() {
        return sortable_candidates_new;
    }

    sortable_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    sortable_candidates_new.into_iter().take(top).collect()
}


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;    
    Ok(())
}
