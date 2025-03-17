use pyo3::{prelude::*, types::PyList};
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};

#[pyfunction]
fn build_set_and_mset(token_ids: Vec<i32>, use_bigrams: bool) -> PyResult<(HashSet<i32>, HashMap<(i32, i32), usize>)> {
    let mut tids_set = HashSet::new();
    let mut mset = HashMap::new();

    if use_bigrams {
        for bigram in token_ids.windows(2) {    
            if bigram.contains(&-1) {
                continue;
            }
            let bigram_tuple = (bigram[0], bigram[1]);
            *mset.entry(bigram_tuple).or_insert(0) += 1;
            tids_set.insert(bigram[0]);
            tids_set.insert(bigram[1]);
        }
    } else {
        for &tid in &token_ids {
            if tid == -1 {
                continue;
            }
            *mset.entry((tid, -1)).or_insert(0) += 1; // use -1 as placeholder for single tokens
            tids_set.insert(tid);
        }
    }

    Ok((tids_set, mset))
}

fn intersection_count<T: Eq + std::hash::Hash + Copy>(
    set1: &HashMap<T, usize>,
    set2: &HashMap<T, usize>,
) -> HashMap<T, usize> {
    set1.iter()
        .filter_map(|(&k, &v1)| set2.get(&k).map(|&v2| (k, v1.min(v2))))
        .collect()
}

#[pyfunction]
fn compute_candidates(
    _py: Python,
    query_tokens: Vec<i32>,
    idx_sets: Bound<'_, PyList>,
    idx_msets: Bound<'_, PyList>,
    matchable_rids: HashSet<usize>,
    top: usize,
    len_legalese: i32,
    high_resemblance_threshold: f64,
    use_bigrams: bool,
) -> PyResult<Vec<(usize, f64)>> {
    let (qset, qmset) = build_set_and_mset(query_tokens, use_bigrams)?;

    let mut candidates = Vec::new();

    for rid in matchable_rids {
        let iset_py = idx_sets.get_item(rid).unwrap();
        let iset: HashSet<i32> = iset_py.extract()?;

        let intersection: HashSet<_> = qset.intersection(&iset).copied().collect();
        if intersection.is_empty() {
            continue;
        }

        let high_intersection: HashSet<_> = intersection.iter().filter(|&&i| i < len_legalese).copied().collect();
        if high_intersection.is_empty() {
            continue;
        }

        let matched_length = intersection.len() as f64;
        let union_len = (qset.len() + iset.len()) as f64 - matched_length;
        let resemblance = matched_length / union_len;

        if resemblance < high_resemblance_threshold {
            continue;
        }

        candidates.push((rid, resemblance));
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    candidates.truncate(top * 10);

    let mut refined_candidates = Vec::new();

    for (rid, _) in candidates {
        let imset_py = idx_msets.get_item(rid).unwrap();
        let imset: HashMap<(i32, i32), usize> = imset_py.extract()?;

        let intersection = intersection_count(&qmset, &imset);
        if intersection.is_empty() {
            continue;
        }

        let matched_length: usize = intersection.values().sum();
        let qlen: usize = qmset.values().sum();
        let ilen: usize = imset.values().sum();
        let union_len = qlen + ilen - matched_length;
        let resemblance = matched_length as f64 / union_len as f64;

        if resemblance >= high_resemblance_threshold {
            refined_candidates.push((rid, resemblance));
        }
    }

    refined_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    refined_candidates.truncate(top);

    Ok(refined_candidates)
}

#[pymodule]
fn candidate_matcher(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_set_and_mset, m)?)?;
    m.add_function(wrap_pyfunction!(compute_candidates, m)?)?;
    Ok(())
}
