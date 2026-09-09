use crate::core::VulkanCore;
use ash::vk;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

///////////////////////////////////////////////////////////////////////////////
// This file was generated with substantial AI assistance and has not been
// carefully reviewed yet. Expect some odd formatting, verbose structure, or
// unclear organization until it gets a manual pass.
///////////////////////////////////////////////////////////////////////////////

pub(crate) struct ProfileLabel(Cow<'static, str>);

impl ProfileLabel {
    fn as_str(&self) -> &str {
        self.0.as_ref()
    }

    fn into_owned(self) -> String {
        self.0.into_owned()
    }
}

impl From<&'static str> for ProfileLabel {
    fn from(label: &'static str) -> Self {
        Self(Cow::Borrowed(label))
    }
}

impl From<String> for ProfileLabel {
    fn from(label: String) -> Self {
        Self(Cow::Owned(label))
    }
}

pub(crate) struct PipelineProfiler {
    query_pool: vk::QueryPool,
    state: RefCell<PipelineProfilerState>,
}

struct PipelineProfilerState {
    queries: Vec<ProfileQuery>,
    query_by_label: HashMap<String, usize>,
}

struct ProfileQuery {
    label: String,
    query: u32,
    pending: bool,
    accum_ms: f32,
    samples: u32,
}

impl PipelineProfiler {
    pub(crate) const REPORT_SAMPLES: u32 = 300;
    const MAX_PROFILE_LABELS: usize = 64;

    pub(crate) unsafe fn new(core: &VulkanCore) -> Self {
        let query_pool = core
            .device
            .create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count((Self::MAX_PROFILE_LABELS * 2) as u32),
                None,
            )
            .unwrap();

        Self {
            query_pool,
            state: RefCell::new(PipelineProfilerState { queries: Vec::new(), query_by_label: HashMap::new() }),
        }
    }

    pub(crate) unsafe fn begin<T>(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        label: impl Into<ProfileLabel>,
        f: impl FnOnce() -> T,
    ) -> T {
        let query = {
            let label = label.into();
            let mut state = self.state.borrow_mut();
            if let Some(&query_index) = state.query_by_label.get(label.as_str()) {
                let query = &mut state.queries[query_index];
                query.pending = true;
                query.query
            } else {
                let query_index = state.queries.len();
                assert!(
                    query_index < Self::MAX_PROFILE_LABELS,
                    "profile query capacity exceeded; increase MAX_PROFILE_LABELS"
                );

                let label = label.into_owned();
                let query = (query_index * 2) as u32;
                state.query_by_label.insert(label.clone(), query_index);
                state.queries.push(ProfileQuery { label, query, pending: true, accum_ms: 0.0, samples: 0 });
                query
            }
        };

        device.cmd_reset_query_pool(cmd, self.query_pool, query, 2);
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.query_pool, query);

        let result = f();

        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.query_pool, query + 1);
        result
    }

    pub(crate) unsafe fn read_and_accumulate(&self, device: &ash::Device, timestamp_period: f32) -> bool {
        let pending_queries: Vec<_> = {
            let state = self.state.borrow();
            state
                .queries
                .iter()
                .enumerate()
                .filter(|(_, query)| query.pending)
                .map(|(index, query)| (index, query.query))
                .collect()
        };

        if pending_queries.is_empty() {
            return false;
        }

        let to_ms = timestamp_period / 1_000_000.0;
        let mut state = self.state.borrow_mut();
        for (index, query) in pending_queries {
            let mut data = [0u64; 2];
            device.get_query_pool_results(self.query_pool, query, &mut data, vk::QueryResultFlags::TYPE_64).unwrap();

            let query = &mut state.queries[index];
            query.accum_ms += to_ms * (data[1] - data[0]) as f32;
            query.samples += 1;
            query.pending = false;
        }

        true
    }

    pub(crate) fn print_report<'a>(profilers: impl IntoIterator<Item = &'a Self>) {
        let mut entries: Vec<ProfileReportEntry> = Vec::new();
        for profiler in profilers {
            let mut state = profiler.state.borrow_mut();
            for query in &mut state.queries {
                if query.samples == 0 {
                    continue;
                }

                if let Some(entry) = entries.iter_mut().find(|entry| entry.label == query.label) {
                    entry.accum_ms += query.accum_ms;
                    entry.samples += query.samples;
                } else {
                    entries.push(ProfileReportEntry {
                        label: query.label.clone(),
                        accum_ms: query.accum_ms,
                        samples: query.samples,
                    });
                }

                query.accum_ms = 0.0;
                query.samples = 0;
            }
        }

        if entries.is_empty() {
            return;
        }

        let mut sum_ms = 0.0;
        let mut lines = Vec::new();
        for entry in entries {
            let average_ms = entry.accum_ms / entry.samples as f32;
            sum_ms += average_ms;
            lines.push(format!("  {} = {average_ms:.4}ms", entry.label));
        }

        println!("timestamp: sum = {sum_ms:.4}ms\n{}", lines.join("\n"));
    }
}

struct ProfileReportEntry {
    label: String,
    accum_ms: f32,
    samples: u32,
}
