use pyo3::{prelude::*, types::{PyDict, PyList}};
use std::collections::HashMap;

// Define a struct to hold the ScoresVector
#[pyclass]
struct ScoresVector {
    is_highly_resemblant: bool,
    containment: f64,
    resemblance: f64,
    matched_length: f64,
}

// Implement the ScoresVector
#[pymethods]
impl ScoresVector {
    #[new]
    fn new(
        is_highly_resemblant: bool,
        containment: f64,
        resemblance: f64,
        matched_length: f64,
    ) -> Self {
        ScoresVector {
            is_highly_resemblant,
            containment,
            resemblance,
            matched_length,
        }
    }
}

// Define a function to compute the intersection of two sets
fn tids_sets_intersector(qset: &Vec<i64>, iset: &Vec<i64>) -> Vec<i64> {
    qset.iter().filter(|x| iset.contains(x)).cloned().collect()
}

// Define a function to compute the intersection of two multisets
fn multisets_intersector(qmset: &HashMap<i64, i64>, imset: &HashMap<i64, i64>) -> HashMap<i64, i64> {
    let mut intersection = HashMap::new();
    for (key, s1count) in qmset {
        if let Some(s2count) = imset.get(key) {
            intersection.insert(*key, std::cmp::min(*s1count, *s2count));
        }
    }
    intersection
}

// Define a function to count the elements in a set
fn tids_set_counter(tids_set: &Vec<i64>) -> i64 {
    tids_set.len() as i64
}

// Define a function to count the elements in a multiset
fn multiset_counter(mset: &HashMap<i64, i64>) -> i64 {
    mset.values().sum::<i64>()
}

// Define a function to filter a set to only include legalese tokens
fn high_tids_set_subset(tids_set: &Vec<i64>, len_legalese: i64) -> Vec<i64> {
    tids_set.iter().filter(|x| **x < len_legalese).cloned().collect()
}

// Define a function to filter a multiset to only include legalese tokens
fn high_tids_multiset_subset(mset: &HashMap<i64, i64>, len_legalese: i64) -> HashMap<i64, i64> {
    let mut subset = HashMap::new();
    for (tid, count) in mset {
        if *tid < len_legalese {
            subset.insert(*tid, *count);
        }
    }
    subset
}

// Define a function to build a set and multiset from a sequence of token ids
fn build_set_and_mset(token_ids: &Vec<i64>) -> (Vec<i64>, HashMap<i64, i64>) {
    let mut tids_mset = HashMap::new();
    let mut tids_set = Vec::new();

    for tid in token_ids {
        if *tid != -1 {
            *tids_mset.entry(*tid).or_insert(0) += 1;
            if !tids_set.contains(tid) {
                tids_set.push(*tid);
            }
        }
    }

    (tids_set, tids_mset)
}

// Define a function to compare two token sets
fn compare_token_sets(
    qset: &Vec<i64>,
    iset: &Vec<i64>,
    intersector: fn(&Vec<i64>, &Vec<i64>) -> Vec<i64>,
    counter: fn(&Vec<i64>) -> i64,
    high_intersection_filter: fn(&Vec<i64>, i64) -> Vec<i64>,
    len_legalese: i64,
    unique: bool,
    rule: Bound<'_, PyDict>,
    filter_non_matching: bool,
    high_resemblance_threshold: f64,
) -> Option<(ScoresVector, ScoresVector, Vec<i64>)> {
    let intersection = intersector(qset, iset);
    if intersection.is_empty() {
        return None;
    }

    let high_intersection = high_intersection_filter(&intersection, len_legalese);

    if filter_non_matching && high_intersection.is_empty() {
        return None;
    }

    let matched_length = counter(&intersection);
    let min_matched_length = rule.get_item("min_matched_length").unwrap().extract::<i64>().unwrap();

    if filter_non_matching && matched_length < min_matched_length {
        return None;
    }

    let iset_len = rule.get_item("length").unwrap().extract::<i64>().unwrap();
    let qset_len = counter(qset);
    let union_len = qset_len + iset_len - matched_length;
    let resemblance = matched_length as f64 / union_len as f64;
    let containment = matched_length as f64 / iset_len as f64;
    let amplified_resemblance = resemblance.powi(2);

    let minimum_containment = rule.get_item("minimum_containment").unwrap().extract::<f64>().unwrap();

    if filter_non_matching && minimum_containment > 0.0 && containment < minimum_containment {
        return None;
    }

    let scores = (
        ScoresVector::new(
            resemblance >= high_resemblance_threshold,
            containment,
            amplified_resemblance,
            matched_length as f64 / 20.0,
        ),
        ScoresVector::new(
            resemblance >= high_resemblance_threshold,
            containment,
            amplified_resemblance,
            matched_length as f64,
        ),
    );

    Some((scores.0, scores.1, intersection))
}

// Define the compute_candidates function
#[pyfunction]
fn compute_candidates(
    py: Python,
    matchable_tokens: Vec<i64>,
    idx: Bound<'_, PyDict>,
    matchable_rids: Vec<i64>,
    top: i64,
    high_resemblance: bool,
    high_resemblance_threshold: f64,
    _use_bigrams: bool,
) -> Vec<(ScoresVector, ScoresVector, i64, PyDict)> {
    let (qset, qmset) = build_set_and_mset(&matchable_tokens);

    let len_legalese = idx.get_item("len_legalese").unwrap().unwrap().extract::<i64>().unwrap();

    let mut sortable_candidates = Vec::new();

    let sets_by_rid = idx.get_item("sets_by_rid").unwrap().unwrap().extract::<Vec<Vec<i64>>>().unwrap();

    for (rid, rule) in idx.get_item("rules_by_rid").unwrap().unwrap().extract::<Vec<PyDict>>().unwrap().iter().enumerate() {
        if !matchable_rids.contains(&(rid as i64)) {
            continue;
        }

        let scores_vectors = compare_token_sets(
            &qset,
            &sets_by_rid[rid],
            tids_sets_intersector,
            tids_set_counter,
            high_tids_set_subset,
            len_legalese,
            true,
            rule,
            true,
            high_resemblance_threshold,
        );

        if let Some((scores_vectors, _)) = scores_vectors {
            let (svr, svf) = scores_vectors;
            if !(high_resemblance && !(svr.is_highly_resemblant && svf.is_highly_resemblant)) {
                sortable_candidates.push((svr, svf, rid as i64, rule.clone()));
            }
        }
    }

    sortable_candidates.sort_by(|a, b| {
        let (svr1, svf1, _, _) = a;
        let (svr2, svf2, _, _) = b;
        svr1.containment.partial_cmp(&svr2.containment).unwrap()
    });

    let mut candidates = Vec::new();

    let msets_by_rid = idx.get_item("msets_by_rid").unwrap().extract::<Vec<HashMap<i64, i64>>>().unwrap();

    for (scores_vectors, rid, rule, _) in sortable_candidates.into_iter().take(top as usize * 10) {
        let scores_vectors = compare_token_sets(
            &qmset.keys().cloned().collect(),
            &msets_by_rid[rid as usize].keys().cloned().collect(),
            multisets_intersector,
            multiset_counter,
            high_tids_multiset_subset,
            len_legalese,
            false,
            rule,
            false,
            high_resemblance_threshold,
        );

        if let Some((scores_vectors, _)) = scores_vectors {
            let (svr, svf) = scores_vectors;
            if !(high_resemblance && !(svr.is_highly_resemblant && svf.is_highly_resemblant)) {
                candidates.push((svr, svf, rid, rule));
            }
        }
    }

    candidates.sort_by(|a, b| {
        let (svr1, svf1, _, _) = a;
        let (svr2, svf2, _, _) = b;
        svr1.containment.partial_cmp(&svr2.containment).unwrap()
    });

    candidates.into_iter().take(top as usize).collect()
}

// Define the module
#[pymodule]
fn token_sets(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ScoresVector>()?;
    m.add_function(wrap_pyfunction!(compute_candidates, m)?)?;
    Ok(())
}
