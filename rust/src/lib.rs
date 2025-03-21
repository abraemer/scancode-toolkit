use pyo3::{
    prelude::*,
    types::{PyList},
};
use std::{
    collections::{HashMap, HashSet},
    hash::{BuildHasherDefault, DefaultHasher},
};

// maturin develop --release -- --cfg pyo3_disable_reference_pool --cfg pyo3_leak_on_drop_without_reference_pool

struct RuleInfo {
    min_matched_length_unique: u64,
    min_matched_length: u64,
    min_high_matched_length_unique: u64,
    min_high_matched_length: u64,
    minimum_containment: f64,
}
static mut INITIALIZED: bool = false;
static mut RULES_BY_RID: Vec<RuleInfo> = Vec::new();
static mut SETS_BY_RID: Vec<Option<HashSet<i64>>> = Vec::new();
static mut MSETS_BY_RID: Vec<Option<HashMap<i64, usize>>> = Vec::new();
static mut MATCHABLE_RIDS: HashSet<usize, BuildHasherDefault<DefaultHasher>> =
    HashSet::with_hasher(BuildHasherDefault::new());

#[pyfunction]
fn init_globals(
    _py: Python,
    rules_by_rid: Bound<'_, PyList>,
    sets_by_rid: Bound<'_, PyList>,
    msets_by_rid: Bound<'_, PyList>,
    approx_matchable_rids: HashSet<usize, BuildHasherDefault<DefaultHasher>>,
) {
    let rules: Vec<RuleInfo> = rules_by_rid.iter().map(create_ruleinfo).collect();
    let sets: Vec<Option<HashSet<i64>>> = sets_by_rid.iter().map(unwrap_set).collect();
    let msets: Vec<Option<HashMap<i64, usize>>> = msets_by_rid.iter().map(unwrap_hashmap).collect();
    {
        let num_sets: i64 = sets
            .iter()
            .map(|x| match x {
                Some(a) => 1,
                None => 0,
            })
            .sum();
        let len_sets = sets.len();
        println!("Sets: {num_sets} non-null out of {len_sets} total");
    }
    {
        let num_sets: i64 = msets
            .iter()
            .map(|x| match x {
                Some(a) => 1,
                None => 0,
            })
            .sum();
        let len_sets = msets.len();
        println!("MSets: {num_sets} non-null out of {len_sets} total");
    }
    unsafe {
        RULES_BY_RID = rules;
        MATCHABLE_RIDS = approx_matchable_rids;
        SETS_BY_RID = sets;
        MSETS_BY_RID = msets;
        INITIALIZED = true;
    }
}

fn create_ruleinfo(data: Bound<'_, PyAny>) -> RuleInfo {
    let min_matched_length_unique = data
        .call_method("get_min_matched_length", (true,), None)
        .unwrap();
    let min_matched_length = data
        .call_method("get_min_matched_length", (false,), None)
        .unwrap();
    let min_high_matched_length_unique = data
        .call_method("get_min_high_matched_length", (true,), None)
        .unwrap();
    let min_high_matched_length = data
        .call_method("get_min_high_matched_length", (false,), None)
        .unwrap();
    let minimum_containment = data.getattr("_minimum_containment").unwrap();
    RuleInfo {
        min_matched_length_unique: min_matched_length_unique.extract::<u64>().expect(""),
        min_matched_length: min_matched_length.extract::<u64>().expect(""),
        min_high_matched_length_unique: min_high_matched_length_unique.extract::<u64>().expect(""),
        min_high_matched_length: min_high_matched_length.extract::<u64>().expect(""),
        minimum_containment: minimum_containment.extract::<f64>().expect(""),
    }
}

fn unwrap_set(data: Bound<'_, PyAny>) -> Option<HashSet<i64>> {
    match data.extract::<HashSet<i64>>() {
        Ok(set) => Some(set),
        Err(_) => None,
    }
}

fn unwrap_hashmap(data: Bound<'_, PyAny>) -> Option<HashMap<i64, usize>> {
    match data.extract::<HashMap<i64, usize>>() {
        Ok(mset) => Some(mset),
        Err(_) => None,
    }
}

fn build_set_and_mset(token_ids: Vec<i64>) -> (HashSet<i64>, HashMap<i64, usize>) {
    let mut tids_set = HashSet::new();
    let mut mset: HashMap<i64, usize> = HashMap::new();

    for &tid in &token_ids {
        if tid == -1 {
            continue;
        }
        *mset.entry(tid).or_insert(0) += 1; // use -1 as placeholder for single tokens
        tids_set.insert(tid);
    }

    (tids_set, mset)
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
unsafe fn compute_candidates(
    _py: Python,
    query_tokens: Vec<i64>,
    top: usize,
    len_legalese: i64,
    high_resemblance_threshold: f64,
) -> PyResult<Vec<(usize, f64)>> {
    assert!(INITIALIZED, "Not initialized!");

    let (qset, qmset) = build_set_and_mset(query_tokens);

    let matchable_rids = &MATCHABLE_RIDS;
    let idx_sets = &SETS_BY_RID;
    let idx_msets = &MSETS_BY_RID;

    let mut candidates = Vec::new();

    for rid in matchable_rids {
        // let iset_py = idx_sets.get_item(rid).unwrap();
        // let iset: HashSet<i32> = iset_py.extract()?;
        let iset1 = idx_sets.get(*rid).unwrap();
        let iset = iset1.as_ref().unwrap();

        let intersection: HashSet<_> = qset.intersection(&iset).copied().collect();
        if intersection.is_empty() {
            continue;
        }

        let high_intersection: HashSet<_> = intersection
            .iter()
            .filter(|&&i| i < len_legalese)
            .copied()
            .collect();
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
        // let imset_py = idx_msets.get_item(rid).unwrap();
        // let imset: HashMap<(i32, i32), usize> = imset_py.extract()?;
        let imset = idx_msets.get(*rid).unwrap().as_ref().unwrap();

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
            refined_candidates.push((*rid, resemblance));
        }
    }

    refined_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    refined_candidates.truncate(top);

    Ok(refined_candidates)
}

#[pymodule]
fn candidate_matcher(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(init_globals, m)?)?;
    Ok(())
}
