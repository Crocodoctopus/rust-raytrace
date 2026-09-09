use crate::buffer::Buffer;
use crate::core::VulkanCore;
use crate::glsl_types::*;
use crate::image::{Image, ImageView};
use crate::mesh::{GpuMesh, Mesh, PendingGpuMesh, load_mesh};
use crate::pipelines::Pipelines;
use crate::profiling::{PipelineProfiler, ProfileLabel};
use crate::rw_queue::{ResourceQueue, WaitStrategy};
use crate::staging::{StagingPool, StagingSpan, Whole};
use crate::swapchain::Swapchain;
use crate::util::wait_semaphores_any_fallback;
use crate::vk_helpers::*;
use crate::world::{Object, World, WorldDiff};
use ash::vk;
use crossbeam::queue::SegQueue;
use glam::*;
use std::collections::{BTreeMap, HashMap};
use std::mem::offset_of;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use threadpool::ThreadPool;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Debug)]
pub struct HandleCounter(u32);

#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Handle(u32);

impl Iterator for HandleCounter {
    type Item = Handle;
    fn next(&mut self) -> Option<Self::Item> {
        self.0 += 1;
        return Some(Handle(self.0));
    }
}

pub(crate) type MeshHandle = Handle;
pub(crate) type ObjectHandle = Handle;

#[allow(unused)]
const B: u64 = 1;
#[allow(non_upper_case_globals, unused)]
const KiB: u64 = 1024 * B;
#[allow(non_upper_case_globals, unused)]
const MiB: u64 = 1024 * KiB;
#[allow(non_upper_case_globals, unused)]
const GiB: u64 = 1024 * MiB;

pub(crate) const STAGING_ARENA_SIZE: u64 = 512 * MiB;
pub(crate) const STAGING_FIF_BLOCK_SIZE: u64 = 32 * MiB;

pub(crate) const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub(crate) const VISIBILITY_DEPTH: usize = 2;
pub(crate) const VISIBILITY_RESOURCE_QUEUE_LEN: usize = VISIBILITY_DEPTH + MAX_FRAMES_IN_FLIGHT;

// Dedicated HZB/occlusion descriptor set.
pub(crate) const MAX_HZB_DIMENSION: u32 = 8192;
pub(crate) const MAX_HZB_MIPS: u32 = MAX_HZB_DIMENSION.div_ceil(2).ilog2() + 1;
pub(crate) const HZB_SAMPLED_IMAGE_CAPACITY: u32 = 1 + MAX_HZB_MIPS;
pub(crate) const HZB_STORAGE_IMAGE_CAPACITY: u32 = MAX_HZB_MIPS;

/*Generate index
Plan:
0) Data upload
1) frustum_cull
2) render (only visible)
3) build_hzb
4) occlusion_cull
5) render (late visible)
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum PipelineStage {
    DataUpload,
    FrustumCull,
    EarlyDraw,
    BuildHzb,
    OcclusionCull,
    LateDraw,
    FrameEnd,
}

impl PipelineStage {
    const COUNT: usize = Self::FrameEnd as usize + 1;

    const fn wait_value(self, base: u64) -> u64 {
        base as u64 as u64 + self as u64
    }

    const fn signal_value(self, base: u64) -> u64 {
        self.wait_value(base) + 1
    }
}

impl From<PipelineStage> for ProfileLabel {
    fn from(stage: PipelineStage) -> Self {
        match stage {
            PipelineStage::DataUpload => "data_upload",
            PipelineStage::FrustumCull => "frustum_cull",
            PipelineStage::EarlyDraw => "early_draw",
            PipelineStage::BuildHzb => "build_hzb",
            PipelineStage::OcclusionCull => "occlusion_cull",
            PipelineStage::LateDraw => "late_draw",
            PipelineStage::FrameEnd => "frame_end",
        }
        .into()
    }
}

struct SwapchainState {
    // HZB is per-frame scratch, but the ring only needs to be large enough to
    // cover the live visibility window.
    hzb_descriptor_pool: vk::DescriptorPool,
    hzb_images: [Image; MAX_FRAMES_IN_FLIGHT],
    hzb_build_src_views: [Box<[ImageView]>; MAX_FRAMES_IN_FLIGHT],
    hzb_build_dst_views: [Box<[ImageView]>; MAX_FRAMES_IN_FLIGHT],
    hzb_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    hzb_sampler: vk::Sampler,
    overdraw_images: [Image; MAX_FRAMES_IN_FLIGHT],
    overdraw_views: [ImageView; MAX_FRAMES_IN_FLIGHT],

    render_finished: Box<[vk::Semaphore]>,
    image_acquired_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    depth_images: [Image; MAX_FRAMES_IN_FLIGHT],
    depth_views: [ImageView; MAX_FRAMES_IN_FLIGHT],
}

impl SwapchainState {
    unsafe fn new(
        core: &VulkanCore,
        swapchain: &Swapchain,
        pipelines: &Pipelines,
        overdraw_sets: &[vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    ) -> Self {
        let device = &core.device;
        let allocator = &core.allocator;
        let mut hzb_sampler_reduction =
            vk::SamplerReductionModeCreateInfo::default().reduction_mode(vk::SamplerReductionMode::MIN);
        let hzb_sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .push_next(&mut hzb_sampler_reduction)
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .unnormalized_coordinates(false),
                None,
            )
            .unwrap();
        let staging_cmd_buffer = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(core.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];

        device.reset_command_buffer(staging_cmd_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
        device.begin_command_buffer(staging_cmd_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

        let hzb_descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * HZB_SAMPLED_IMAGE_CAPACITY),
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * HZB_STORAGE_IMAGE_CAPACITY),
                    ])
                    .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            )
            .unwrap();
        let hzb_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT] = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(hzb_descriptor_pool)
                    .set_layouts(&[pipelines.hzb_set_layout; MAX_FRAMES_IN_FLIGHT]),
            )
            .unwrap()
            .try_into()
            .unwrap();

        let vk::Extent2D { width, height, .. } = swapchain.extent;
        if width > MAX_HZB_DIMENSION || height > MAX_HZB_DIMENSION {
            panic!("HZB/occlusion descriptor set only supports up to {MAX_HZB_DIMENSION}; got {width}x{height}");
        }

        // Round the half-res base up to the next power of two so the mip chain is regular.
        let hzb_width = width.div_ceil(2).max(1).next_power_of_two();
        let hzb_height = height.div_ceil(2).max(1).next_power_of_two();
        let mipmaps = u32::max(hzb_width, hzb_height).ilog2() + 1;

        if mipmaps > MAX_HZB_MIPS {
            panic!("HZB mip chain exceeds reserved descriptor range: {mipmaps} mips > {MAX_HZB_MIPS}");
        }

        let create_image =
            |extent: vk::Extent2D, format: vk::Format, usage: vk::ImageUsageFlags, mip_levels: u32| -> Image {
                let create_info = image2d_create_info()
                    .extent(extent3d_from_extent2d(extent))
                    .format(format)
                    .usage(usage)
                    .mip_levels(mip_levels);
                let (image, alloc) =
                    vk_mem::Alloc::create_image(allocator, &create_info, &device_local_alloc()).unwrap();
                Image { image, alloc }
            };

        let create_view = |image: vk::Image,
                           format: vk::Format,
                           aspect: vk::ImageAspectFlags,
                           base_mip_level: u32,
                           level_count: u32|
         -> ImageView {
            let view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: aspect,
                            base_mip_level,
                            level_count,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .unwrap();
            ImageView { view }
        };

        let hzb_images = std::array::from_fn(|_| {
            create_image(
                vk::Extent2D { width: hzb_width, height: hzb_height },
                vk::Format::R32_SFLOAT,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
                mipmaps,
            )
        });

        let hzb_build_src_views = std::array::from_fn(|slot| {
            (0..mipmaps)
                .into_iter()
                .map(|level| {
                    create_view(hzb_images[slot].image, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR, level, 1)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });

        let hzb_build_dst_views = std::array::from_fn(|slot| {
            (0..mipmaps)
                .into_iter()
                .map(|level| {
                    create_view(hzb_images[slot].image, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR, level, 1)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });

        let render_finished = (0..swapchain.images.len())
            .into_iter()
            .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
            .collect();

        let image_acquired_semaphores =
            std::array::from_fn(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap());

        let depth_images = std::array::from_fn(|_| {
            create_image(
                swapchain.extent,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                1,
            )
        });

        let depth_views = std::array::from_fn(|i| {
            create_view(depth_images[i].image, vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH, 0, 1)
        });

        let overdraw_images = std::array::from_fn(|_| {
            create_image(
                swapchain.extent,
                vk::Format::R32_UINT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                1,
            )
        });

        let overdraw_views = std::array::from_fn(|i| {
            create_view(overdraw_images[i].image, vk::Format::R32_UINT, vk::ImageAspectFlags::COLOR, 0, 1)
        });

        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            let hzb_src_infos: Box<_> = hzb_build_src_views[slot]
                .iter()
                .map(|image_view| {
                    vk::DescriptorImageInfo::default()
                        .image_view(image_view.view)
                        .sampler(hzb_sampler)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();

            let hzb_dst_infos: Box<_> = hzb_build_dst_views[slot]
                .iter()
                .map(|image_view| {
                    vk::DescriptorImageInfo::default()
                        .image_view(image_view.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                })
                .collect();

            let depth_info = [vk::DescriptorImageInfo::default()
                .image_view(depth_views[slot].view)
                .sampler(hzb_sampler)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];

            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(hzb_sets[slot])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(depth_info.len() as u32)
                        .image_info(&depth_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(hzb_sets[slot])
                        .dst_binding(0)
                        .dst_array_element(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(hzb_src_infos.len() as u32)
                        .image_info(&hzb_src_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(hzb_sets[slot])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(hzb_dst_infos.len() as u32)
                        .image_info(&hzb_dst_infos),
                ],
                &[],
            );
        }

        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            let overdraw_info = [vk::DescriptorImageInfo::default()
                .image_view(overdraw_views[slot].view)
                .image_layout(vk::ImageLayout::GENERAL)];

            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(overdraw_sets[slot])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .image_info(&overdraw_info)],
                &[],
            );
        }

        let mut barriers = Vec::new();
        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .image(hzb_images[slot].image)
                    .subresource_range(COLOR_2D_SUBRESOURCE_RANGE.level_count(vk::REMAINING_MIP_LEVELS))
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL),
            );
        }

        barriers.extend((0..MAX_FRAMES_IN_FLIGHT).into_iter().map(|i| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .image(overdraw_images[i].image)
                .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
        }));

        barriers.extend((0..MAX_FRAMES_IN_FLIGHT).into_iter().map(|i| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .image(depth_images[i].image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
        }));

        device
            .cmd_pipeline_barrier2(staging_cmd_buffer, &vk::DependencyInfo::default().image_memory_barriers(&barriers));

        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            device.cmd_clear_color_image(
                staging_cmd_buffer,
                hzb_images[slot].image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                &[vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mipmaps)
                    .base_array_layer(0)
                    .layer_count(1)],
            );
        }

        let barriers = Vec::from_iter((0..MAX_FRAMES_IN_FLIGHT).map(|slot| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .image(hzb_images[slot].image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(vk::REMAINING_MIP_LEVELS)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        }));

        device
            .cmd_pipeline_barrier2(staging_cmd_buffer, &vk::DependencyInfo::default().image_memory_barriers(&barriers));

        device.end_command_buffer(staging_cmd_buffer).unwrap();
        device
            .queue_submit2(
                *core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default().command_buffer_infos(&[
                    vk::CommandBufferSubmitInfo::default().command_buffer(staging_cmd_buffer)
                ])],
                vk::Fence::null(),
            )
            .unwrap();
        device.queue_wait_idle(*core.graphics_queue.lock().unwrap()).unwrap();
        device.free_command_buffers(core.cmd_pool, &[staging_cmd_buffer]);

        Self {
            hzb_descriptor_pool,
            hzb_images,
            hzb_build_src_views,
            hzb_build_dst_views,
            hzb_sets,
            hzb_sampler,
            overdraw_images,
            overdraw_views,
            render_finished,
            image_acquired_semaphores,
            depth_images,
            depth_views,
        }
    }

    unsafe fn free(self, core: &VulkanCore) {
        let device = &core.device;
        let allocator = &core.allocator;
        let Self {
            hzb_descriptor_pool,
            hzb_images,
            hzb_build_src_views,
            hzb_build_dst_views,
            hzb_sets: _,
            hzb_sampler,
            overdraw_images,
            overdraw_views,
            render_finished,
            image_acquired_semaphores,
            depth_images,
            depth_views,
        } = self;

        device.destroy_sampler(hzb_sampler, None);

        for views in hzb_build_src_views {
            for view in views {
                device.destroy_image_view(view.view, None);
            }
        }
        for views in hzb_build_dst_views {
            for view in views {
                device.destroy_image_view(view.view, None);
            }
        }
        for mut image in hzb_images {
            unsafe { allocator.destroy_image(image.image, &mut image.alloc) }
        }
        for view in overdraw_views {
            device.destroy_image_view(view.view, None);
        }
        for mut image in overdraw_images {
            unsafe { allocator.destroy_image(image.image, &mut image.alloc) }
        }

        for semaphore in render_finished {
            device.destroy_semaphore(semaphore, None);
        }
        for semaphore in image_acquired_semaphores {
            device.destroy_semaphore(semaphore, None);
        }

        for view in depth_views {
            device.destroy_image_view(view.view, None);
        }
        for mut image in depth_images {
            unsafe { allocator.destroy_image(image.image, &mut image.alloc) }
        }

        device.destroy_descriptor_pool(hzb_descriptor_pool, None);
    }
}

struct Fif {
    profiler: PipelineProfiler,

    _descriptor_pool: vk::DescriptorPool,
    _overdraw_descriptor_pool: vk::DescriptorPool,
    cmd_buffers: [vk::CommandBuffer; PipelineStage::COUNT],
    staging_buffer: StagingSpan,
    frame_set: vk::DescriptorSet,
    overdraw_set: vk::DescriptorSet,
    frame_global_buffer: Buffer<GpuFrameGlobal>,

    // The next submission's base is also the completion value of the current
    // submission because stage timeline values are contiguous.
    base: u64,
    timeline: vk::Semaphore,

    // This FIF's snapshot and persistent mesh layout. Objects are uploaded afresh each frame.
    world: World,
    scene_index_offsets: HashMap<MeshHandle, u32>,
    scene_index_buffer: Buffer<[GpuIndex]>,
    object_instance_buffer: Buffer<[GpuObjectInstance]>,
    indirect_cmd_buffer: Buffer<GpuDrawCommandBuffer>,
    frustum_passing_meshlet_buffer: Buffer<GpuFrustumPassingMeshletBuffer>,

    // Renderer owns the visibility columns; this FIF owns their selection for
    // its current submission.
    visibility_read: usize,
    visibility_write: usize,
}

pub struct Renderer {
    //
    core: Arc<VulkanCore>,

    /* Swapchain data: */
    swapchain: Swapchain,

    /* Profiling: */
    profile_report_samples: u32,

    /* Pipelines: */
    pipelines: Pipelines,

    /* Generic resource containers: */
    cwd: PathBuf,
    resource_counter: HandleCounter,
    // Canonical CPU scene. A FIF keeps its own resident, submitted snapshot.
    world: World,
    meshes: BTreeMap<MeshHandle, Arc<Mesh>>,

    // Some cpu -> gpu resources.
    gpu_meshes: HashMap<MeshHandle, GpuMesh>,
    pending_gpu_meshes: HashMap<MeshHandle, PendingGpuMesh>,

    //
    thread_pool: ThreadPool,
    graphics_cmd_pools: Arc<SegQueue<vk::CommandPool>>,

    /* Staging: */
    staging: Arc<StagingPool>,

    /* Frames in flight: */
    fifs: [Fif; MAX_FRAMES_IN_FLIGHT],

    // Global descriptor set.
    global_set: vk::DescriptorSet,

    // Orders uploads across FIFs, including initialization of shared visibility.
    upload_sync_timeline: vk::Semaphore,
    upload_sync_counter: u64,

    // Dirty flags for resource regeneration.
    swapchain_states_dirty: bool,

    // Swapchain management.
    swapchain_states: SwapchainState,

    // Visibility is object-owned temporal state. FIFs reserve shared column
    // indices from this queue for each submission.
    visibility_buffers: HashMap<ObjectHandle, [Buffer<[u32]>; VISIBILITY_RESOURCE_QUEUE_LEN]>,
    visibility_resource_queue: ResourceQueue<usize>,

    // Various render state data.
    frame: usize,
    pub cam_pos: Vec3,
    pub cam_rot: Vec2, // YX
    pub overdraw_enabled: bool,
    pub overshade_enabled: bool,
}

const LOD_DISTANCE_BIAS: f32 = 2.0;
const LOD_DISTANCE_OFFSET: f32 = 0.25;

impl Drop for Renderer {
    fn drop(&mut self) {
        panic!("{} dropped implicitly; call explicit renderer shutdown before drop", std::any::type_name::<Self>());
    }
}

impl Renderer {
    pub fn new(
        cwd: impl AsRef<Path>,
        viewport_w: u32,
        viewport_h: u32,
        display: impl HasDisplayHandle + HasWindowHandle,
    ) -> Self {
        unsafe {
            let core = Arc::new(VulkanCore::new(display));
            let device = &core.device;
            let allocator = &core.allocator;

            // Build swapchain from core.
            let swapchain = Swapchain::new(&core, vk::Extent2D { width: viewport_w, height: viewport_h });

            let pipelines = Pipelines::new(&core);

            // Staging data.
            let staging = Arc::new(StagingPool::new(allocator, STAGING_ARENA_SIZE));
            let thread_pool =
                ThreadPool::new(std::thread::available_parallelism().map_or(1, |parallelism| parallelism.get()));
            let graphics_cmd_pools = Arc::new(SegQueue::new());

            //
            let global_descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1024 + (MAX_FRAMES_IN_FLIGHT as u32 * HZB_SAMPLED_IMAGE_CAPACITY)),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(1024 + (MAX_FRAMES_IN_FLIGHT as u32 * HZB_STORAGE_IMAGE_CAPACITY)),
                        ])
                        .max_sets(1 + MAX_FRAMES_IN_FLIGHT as u32)
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                    None,
                )
                .unwrap();

            let upload_sync_timeline = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(
                        &mut vk::SemaphoreTypeCreateInfo::default()
                            .semaphore_type(vk::SemaphoreType::TIMELINE)
                            .initial_value(0),
                    ),
                    None,
                )
                .unwrap();

            // Descriptor sets.
            let global_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(global_descriptor_pool)
                        .set_layouts(&[pipelines.global_set_layout]),
                )
                .unwrap()[0];

            let fifs = std::array::from_fn(|_| {
                let descriptor_pool = device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&[vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::STORAGE_BUFFER)
                                .descriptor_count(1)])
                            .max_sets(1)
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                        None,
                    )
                    .unwrap();
                let overdraw_descriptor_pool = device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&[
                                vk::DescriptorPoolSize::default()
                                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                                    .descriptor_count(1),
                                vk::DescriptorPoolSize::default()
                                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                                    .descriptor_count(2),
                            ])
                            .max_sets(1)
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                        None,
                    )
                    .unwrap();
                let frame_set = device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(&[pipelines.frame_set_layout]),
                    )
                    .unwrap()[0];
                let overdraw_set = device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(overdraw_descriptor_pool)
                            .set_layouts(&[pipelines.overdraw_set_layout]),
                    )
                    .unwrap()[0];
                let frame_global_buffer = Buffer::<GpuFrameGlobal>::new(
                    &allocator,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    vk_mem::MemoryUsage::AutoPreferDevice,
                );
                let descriptor_buffer_infos = [vk::DescriptorBufferInfo::default()
                    .buffer(frame_global_buffer.vk_handle())
                    .offset(0)
                    .range(vk::WHOLE_SIZE)];

                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(frame_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&descriptor_buffer_infos)],
                    &[],
                );

                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(overdraw_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&descriptor_buffer_infos)],
                    &[],
                );

                let cmd_buffers = device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(core.cmd_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(PipelineStage::COUNT as _),
                    )
                    .unwrap()
                    .try_into()
                    .unwrap();
                let timeline = device
                    .create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(
                            &mut vk::SemaphoreTypeCreateInfo::default()
                                .semaphore_type(vk::SemaphoreType::TIMELINE)
                                .initial_value(0),
                        ),
                        None,
                    )
                    .unwrap();

                Fif {
                    profiler: PipelineProfiler::new(&core),
                    _descriptor_pool: descriptor_pool,
                    _overdraw_descriptor_pool: overdraw_descriptor_pool,
                    cmd_buffers,
                    staging_buffer: staging.alloc(STAGING_FIF_BLOCK_SIZE),
                    frame_set,
                    overdraw_set,
                    frame_global_buffer,
                    base: 0,
                    timeline,
                    world: World::default(),
                    scene_index_offsets: HashMap::new(),
                    scene_index_buffer: Buffer::null(),
                    object_instance_buffer: Buffer::null(),
                    indirect_cmd_buffer: Buffer::null(),
                    frustum_passing_meshlet_buffer: Buffer::null(),
                    visibility_read: 0,
                    visibility_write: 0,
                }
            });

            let overdraw_sets = std::array::from_fn(|fif| fifs[fif].overdraw_set);
            let swapchain_states = SwapchainState::new(&core, &swapchain, &pipelines, &overdraw_sets);

            //
            Self {
                core,

                swapchain,

                profile_report_samples: 0,

                pipelines,

                cwd: cwd.as_ref().to_owned(),
                resource_counter: HandleCounter(0),
                world: World::default(),
                meshes: BTreeMap::new(),

                gpu_meshes: HashMap::new(),
                pending_gpu_meshes: HashMap::new(),
                thread_pool,
                graphics_cmd_pools,

                staging,
                fifs,
                global_set,
                upload_sync_timeline,
                upload_sync_counter: 0,
                swapchain_states_dirty: false,
                visibility_buffers: HashMap::new(),
                visibility_resource_queue: ResourceQueue::new(MAX_FRAMES_IN_FLIGHT, 0..VISIBILITY_RESOURCE_QUEUE_LEN),
                swapchain_states,

                frame: 0,
                cam_pos: Vec3::new(0., 0., 3.),
                cam_rot: <_>::default(),
                overdraw_enabled: false,
                overshade_enabled: false,
            }
        }
    }

    fn rebuild_swapchain_states_if_dirty(&mut self) {
        if self.swapchain_states_dirty {
            self.swapchain_states_dirty = false;
            let timelines = self.fifs.each_ref().map(|fif| fif.timeline);
            let bases = self.fifs.each_ref().map(|fif| fif.base);
            unsafe {
                self.core
                    .device
                    .wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&timelines).values(&bases), u64::MAX)
                    .unwrap();
            }

            let mut swapchain = unsafe { Swapchain::new(&self.core, self.swapchain.extent) };
            let overdraw_sets = self.fifs.each_ref().map(|fif| fif.overdraw_set);
            let mut swapchain_states =
                unsafe { SwapchainState::new(&self.core, &swapchain, &self.pipelines, &overdraw_sets) };
            std::mem::swap(&mut self.swapchain_states, &mut swapchain_states);
            unsafe {
                swapchain_states.free(&self.core);
            }
            std::mem::swap(&mut self.swapchain, &mut swapchain);
            unsafe {
                swapchain.free(&self.core.device);
            }
        }
    }

    fn promote_completed_gpu_meshes(&mut self) {
        for (mesh_id, pending) in std::mem::take(&mut self.pending_gpu_meshes) {
            match unsafe { pending.try_unwrap(&self.core) } {
                Ok(gpu_mesh) => {
                    self.gpu_meshes.insert(mesh_id, gpu_mesh);
                }
                Err(pending) => {
                    self.pending_gpu_meshes.insert(mesh_id, pending);
                }
            }
        }
    }

    fn upload_missing_gpu_meshes(&mut self) {
        let missing_meshes: Vec<_> =
            self.world.meshes.iter().filter(|mesh_id| !self.gpu_meshes.contains_key(mesh_id)).copied().collect();
        for mesh_id in missing_meshes {
            if self.pending_gpu_meshes.contains_key(&mesh_id) {
                continue;
            }
            let mesh = Arc::clone(self.meshes.get(&mesh_id).unwrap());
            let pending = unsafe {
                PendingGpuMesh::submit(&self.thread_pool, &self.graphics_cmd_pools, &self.core, &self.staging, mesh)
            };
            self.pending_gpu_meshes.insert(mesh_id, pending);
        }
    }

    fn reserve_available_frame_slot(&mut self) -> (usize, u64) {
        let timelines = self.fifs.each_ref().map(|fif| fif.timeline);
        let bases = self.fifs.each_ref().map(|fif| fif.base);
        unsafe {
            wait_semaphores_any_fallback(&self.core.device, &timelines, &bases).unwrap();
        }

        let index = self
            .fifs
            .iter()
            .enumerate()
            .find(|(_, fif)| unsafe { self.core.device.get_semaphore_counter_value(fif.timeline).unwrap() == fif.base })
            .unwrap()
            .0;

        let base = self.fifs[index].base;
        self.fifs[index].base += PipelineStage::COUNT as u64;
        (index, base)
    }

    unsafe fn read_and_accumulate_frame_profile(&mut self, frame_index: usize) {
        let read_profile = self.fifs[frame_index]
            .profiler
            .read_and_accumulate(&self.core.device, self.core.physical_device_properties.limits.timestamp_period);
        if !read_profile {
            return;
        }

        self.profile_report_samples += 1;
        if self.profile_report_samples == PipelineProfiler::REPORT_SAMPLES {
            PipelineProfiler::print_report(self.fifs.iter().map(|fif| &fif.profiler));
            self.profile_report_samples = 0;
        }
    }

    unsafe fn record_and_submit_data_upload_stage(
        &mut self,
        frame_index: usize,
        frame_timeline_base: u64,
        pipeline_semaphore: vk::Semaphore,
        world: &World,
        object_dispatch: &mut Vec<(u16, u16)>,
    ) -> Vec<vk::SemaphoreSubmitInfo<'static>> {
        let Fif {
            profiler,
            cmd_buffers,
            staging_buffer,
            frame_global_buffer,
            visibility_read,
            visibility_write,
            scene_index_offsets,
            scene_index_buffer,
            object_instance_buffer,
            indirect_cmd_buffer,
            frustum_passing_meshlet_buffer,
            ..
        } = &mut self.fifs[frame_index];
        let data_upload = cmd_buffers[PipelineStage::DataUpload as usize];
        let upload_sync_wait_value = self.upload_sync_counter;
        self.upload_sync_counter += 1;
        let upload_sync_signal_value = self.upload_sync_counter;
        let swapchain_extent = self.swapchain.extent;
        let visibility_wait_strategy = WaitStrategy {
            semaphore: pipeline_semaphore,
            value: PipelineStage::OcclusionCull.signal_value(frame_timeline_base),
        };

        *visibility_read = *self.visibility_resource_queue.read(&self.core.device, visibility_wait_strategy);
        let (column, visibility_resource_waits) =
            self.visibility_resource_queue.write(&self.core.device, frame_index, visibility_wait_strategy).unwrap();
        *visibility_write = *column;

        object_dispatch.reserve(world.objects.len());
        let camera_forward = Vec3::new(
            self.cam_rot[0].sin() * self.cam_rot[1].cos(),
            self.cam_rot[1].sin(),
            -self.cam_rot[0].cos() * self.cam_rot[1].cos(),
        );
        let mut object_data = Vec::with_capacity(world.objects.len());
        let mut new_visibility_buffers = Vec::new();
        for (handle, obj) in &world.objects {
            let mesh = self.meshes.get(&obj.mesh).unwrap();
            let gpu_mesh = self.gpu_meshes.get(&obj.mesh).unwrap();
            let scene_index_offset = *scene_index_offsets.get(&obj.mesh).unwrap();

            let distance = (self.cam_pos - obj.position).length();
            let object_radius = obj.scale * mesh.scale * mesh.radius;
            let lod_ratio =
                ((distance.max(1e-5) / object_radius.max(1e-5)) - LOD_DISTANCE_OFFSET).max(1e-5) * LOD_DISTANCE_BIAS;
            let lod_id = lod_ratio.log2().floor().clamp(0.0, (mesh.lod_count.saturating_sub(1)) as f32) as u8;
            let meshlet_subrange = gpu_mesh.meshlet_lod_to_offset[&lod_id.min(mesh.lod_count)].clone();

            object_dispatch.push((object_data.len() as u16, meshlet_subrange.end - meshlet_subrange.start));

            // New objects have independent history allocations. Initialization is submitted
            // through the shared upload timeline before any FIF can consume them.
            let object_visibility_buffers = self.visibility_buffers.entry(*handle).or_insert_with(|| {
                let len = mesh.lods.iter().map(|lod| lod.len() as u32).max().unwrap_or(0).max(1);
                std::array::from_fn(|_| {
                    let buffer = Buffer::<[u32]>::new(
                        &self.core.allocator,
                        len,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                        vk_mem::MemoryUsage::AutoPreferDevice,
                    );
                    new_visibility_buffers.push((buffer.vk_handle(), buffer.size() as u64));
                    buffer
                })
            });
            let visibility_buffer = object_visibility_buffers[*visibility_write].vk_handle();
            let previous_visibility_buffer = object_visibility_buffers[*visibility_read].vk_handle();

            object_data.push(GpuObjectInstance {
                position: obj.position,
                scale: obj.scale * mesh.scale,
                orientation: obj.orientation,
                vertex_buffer: self.core.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(gpu_mesh.vertex_buffer.vk_handle()),
                ),
                // This BDA is corrected for LOD subrange.
                meshlet_buffer: self.core.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(gpu_mesh.meshlet_buffer.vk_handle()),
                ) + meshlet_subrange.start as u64 * std::mem::size_of::<GpuMeshlet>() as u64,
                visibility_buffer: self
                    .core
                    .device
                    .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(visibility_buffer)),
                previous_visibility_buffer: self.core.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(previous_visibility_buffer),
                ),
                texture_id: 0,
                scene_index_offset,
            });
        }

        // TODO: consider gpu sorting?
        object_dispatch.sort_unstable_by(|a, b| {
            let a_pos = object_data[usize::from(a.0)].position;
            let b_pos = object_data[usize::from(b.0)].position;
            let a_distance = (self.cam_pos - a_pos).length_squared();
            let b_distance = (self.cam_pos - b_pos).length_squared();
            a_distance.total_cmp(&b_distance)
        });

        // Reverse-Z projection: near maps to 1.0, infinity tends toward 0.0.
        let projection = Mat4::perspective_infinite_reverse_rh(
            std::f32::consts::FRAC_PI_6,
            swapchain_extent.width as f32 / swapchain_extent.height as f32,
            0.1,
        );

        let p = camera_forward;
        let view = Mat4::look_to_rh(self.cam_pos, p, Vec3::new(0., 1., 0.));

        // Frustum plane data.
        let normalize_plane = |p: Vec4| p / p.xyz().length();
        let temp = projection.transpose();
        let frustum_x = normalize_plane(temp.w_axis + temp.x_axis);
        let frustum_y = normalize_plane(temp.w_axis + temp.y_axis);
        let frustum = Vec4::from([frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z]);

        let frame_global = GpuFrameGlobal {
            pv: projection * view,
            proj: projection,
            view,
            camera_position: self.cam_pos.extend(1.0),
            camera_direction: p.extend(0.0),
            light_position: Vec4::new(1.0, 0.0, 0.0, 1.0),
            light_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            frustum,
            screen_info: Vec4::new(swapchain_extent.width as f32, swapchain_extent.height as f32, 0.0, 0.0),
            draw_cmd_buffer: self.core.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(indirect_cmd_buffer.vk_handle()),
            ),
            object_buffer: self.core.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(object_instance_buffer.vk_handle()),
            ),
            frustum_passing_meshlet_buffer: self.core.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(frustum_passing_meshlet_buffer.vk_handle()),
            ),
            occlusion_dispatch: vk::DispatchIndirectCommand { x: 0, y: 1, z: 1 },
        };

        record_cmd_buffer(&self.core.device, data_upload, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::DataUpload, || {
                // TODO: Record only the index copies selected by reconciliation's world diff.
                // Until then, populate the complete target layout on every submission.
                let barriers: Vec<_> = world
                    .meshes
                    .iter()
                    .map(|mesh_id| {
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                            .buffer(self.gpu_meshes[mesh_id].index_buffer.vk_handle())
                            .offset(0)
                            .size(vk::WHOLE_SIZE)
                    })
                    .collect();
                self.core
                    .device
                    .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().buffer_memory_barriers(&barriers));
                for mesh_id in &world.meshes {
                    let source = &self.gpu_meshes[mesh_id].index_buffer;
                    if source.size() == 0 {
                        continue;
                    }
                    self.core.device.cmd_copy_buffer(
                        cmd,
                        source.vk_handle(),
                        scene_index_buffer.vk_handle(),
                        &[vk::BufferCopy::default()
                            .dst_offset(scene_index_offsets[mesh_id] as u64 * std::mem::size_of::<GpuIndex>() as u64)
                            .size(source.size() as u64)],
                    );
                }

                for (buffer, size) in new_visibility_buffers {
                    self.core.device.cmd_fill_buffer(cmd, buffer, 0, size, 0);
                }

                // CPU object records are rebuilt every frame and uploaded directly into this FIF.
                staging_buffer.reset();
                if !object_data.is_empty() {
                    self.staging.stage(
                        staging_buffer,
                        &self.core.device,
                        cmd,
                        object_instance_buffer,
                        Whole(object_data),
                    );
                }
                self.staging.stage(staging_buffer, &self.core.device, cmd, frame_global_buffer, Whole(frame_global));

                // Set indirect & frustum_passing lens to 0.
                self.core.device.cmd_fill_buffer(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    std::mem::size_of::<u32>() as u64,
                    0,
                );
                self.core.device.cmd_fill_buffer(
                    cmd,
                    frustum_passing_meshlet_buffer.vk_handle(),
                    0,
                    std::mem::size_of::<u32>() as u64,
                    0,
                );
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(data_upload)])
                    .wait_semaphore_infos(&[
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::DataUpload.wait_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER),
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.upload_sync_timeline)
                            .value(upload_sync_wait_value)
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER),
                    ])
                    .signal_semaphore_infos(&[
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::DataUpload.signal_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER),
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.upload_sync_timeline)
                            .value(upload_sync_signal_value)
                            .stage_mask(vk::PipelineStageFlags2::TRANSFER),
                    ])],
                vk::Fence::null(),
            )
            .unwrap();

        visibility_resource_waits
    }

    unsafe fn record_and_submit_frustum_cull_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        pipeline_semaphore: vk::Semaphore,
        object_dispatch: Vec<(u16, u16)>,
        visibility_resource_waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    ) {
        let fif = &self.fifs[frame_index];
        let profiler = &fif.profiler;
        let frustum_cull = fif.cmd_buffers[PipelineStage::FrustumCull as usize];
        let frame_set = fif.frame_set;

        record_cmd_buffer(&self.core.device, frustum_cull, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::FrustumCull, || {
                self.core.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.frustum_cull_pipeline_layout,
                    0,
                    &[frame_set],
                    &[],
                );

                self.core.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.frustum_cull_pipeline,
                );

                for (object_index, meshlet_count) in object_dispatch {
                    let push_constants = u32::from(object_index) | (u32::from(meshlet_count) << 16);
                    self.core.device.cmd_push_constants(
                        cmd,
                        self.pipelines.frustum_cull_pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &push_constants.to_ne_bytes(),
                    );

                    self.core.device.cmd_dispatch(cmd, (meshlet_count as u32).div_ceil(64), 1, 1);
                }
            })
        });

        let frustum_cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(frustum_cull)];
        let frustum_signal_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(pipeline_semaphore)
            .value(PipelineStage::FrustumCull.signal_value(frame_timeline_base))
            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)];
        let mut frustum_waits = Vec::with_capacity(2 + visibility_resource_waits.len());
        frustum_waits.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(pipeline_semaphore)
                .value(PipelineStage::FrustumCull.wait_value(frame_timeline_base))
                .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
        );
        frustum_waits.extend(visibility_resource_waits);
        let frustum_submit_infos = [vk::SubmitInfo2::default()
            .command_buffer_infos(&frustum_cmd_infos)
            .wait_semaphore_infos(&frustum_waits)
            .signal_semaphore_infos(&frustum_signal_infos)];
        self.core
            .device
            .queue_submit2(*self.core.graphics_queue.lock().unwrap(), &frustum_submit_infos, vk::Fence::null())
            .unwrap();
    }

    unsafe fn record_and_submit_early_draw_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        image_index: u32,
        pipeline_semaphore: vk::Semaphore,
        image_acquired: vk::Semaphore,
        debug_draw_enabled: bool,
    ) {
        let fif = &self.fifs[frame_index];
        let profiler = &fif.profiler;
        let early_draw = fif.cmd_buffers[PipelineStage::EarlyDraw as usize];
        let swapchain_extent = self.swapchain.extent;
        let swapchain_image = self.swapchain.images[image_index as usize];
        let swapchain_view = self.swapchain.views[image_index as usize];
        let depth_view = &self.swapchain_states.depth_views[frame_index];
        let overdraw_image = &self.swapchain_states.overdraw_images[frame_index];
        let global_set = self.global_set;
        let frame_set = fif.frame_set;
        let overdraw_set = fif.overdraw_set;
        let overshade_enabled = self.overshade_enabled;
        let scene_index_buffer = &fif.scene_index_buffer;
        let indirect_cmd_buffer = &fif.indirect_cmd_buffer;

        if debug_draw_enabled {
            let swapchain_info = [vk::DescriptorImageInfo::default()
                .image_view(swapchain_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            self.core.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(overdraw_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(swapchain_info.len() as u32)
                    .image_info(&swapchain_info)],
                &[],
            );
        }

        record_cmd_buffer(&self.core.device, early_draw, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::EarlyDraw, || {
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        // Reverse-Z clears to the farthest depth value.
                        depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                    });

                let render_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: swapchain_extent.width,
                            height: swapchain_extent.height,
                        },
                    })
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                if debug_draw_enabled {
                    self.core.device.cmd_clear_color_image(
                        cmd,
                        overdraw_image.image,
                        vk::ImageLayout::GENERAL,
                        &vk::ClearColorValue { uint32: [0, 0, 0, 0] },
                        &[vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)],
                    );

                    // The overdraw count image was cleared by transfer; make it available to fragment shader atomics.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .image(overdraw_image.image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );

                    self.core.device.cmd_begin_rendering(cmd, &render_info.color_attachments(&[]));
                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.overdraw_render_pipeline_layout,
                        0,
                        &[global_set, overdraw_set],
                        &[],
                    );
                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        if overshade_enabled {
                            self.pipelines.overshade_render_pipeline
                        } else {
                            self.pipelines.overdraw_render_pipeline
                        },
                    );
                } else {
                    // Swapchain image must move from presentable usage to color attachment usage for the normal render
                    // path.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)]),
                    );

                    self.core.device.cmd_begin_rendering(
                        cmd,
                        &render_info.color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(swapchain_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                            })]),
                    );

                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.render_pipeline_layout,
                        0,
                        &[global_set, frame_set],
                        &[],
                    );

                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.render_pipeline,
                    );
                }

                self.core.device.cmd_set_viewport(
                    cmd,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: swapchain_extent.width as f32,
                        height: swapchain_extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );

                self.core.device.cmd_set_scissor(
                    cmd,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_extent,
                    }],
                );

                self.core.device.cmd_bind_index_buffer(cmd, scene_index_buffer.vk_handle(), 0, vk::IndexType::UINT32);

                self.core.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    std::mem::size_of::<GpuIndex>() as u64,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    indirect_cmd_buffer.len(),
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                self.core.device.cmd_end_rendering(cmd);
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(early_draw)])
                    .wait_semaphore_infos(&[
                        // We can delay this wait until late_draw when overdraw mode is on, but its probably not a
                        // useful optimization.
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::EarlyDraw.wait_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT),
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(image_acquired)
                            .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
                    ])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::EarlyDraw.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_build_hzb_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        pipeline_semaphore: vk::Semaphore,
    ) {
        let fif = &self.fifs[frame_index];
        let profiler = &fif.profiler;
        let build_hzb = fif.cmd_buffers[PipelineStage::BuildHzb as usize];
        let swapchain_extent = self.swapchain.extent;
        let hzb_base_width = swapchain_extent.width.div_ceil(2).max(1);
        let hzb_base_height = swapchain_extent.height.div_ceil(2).max(1);
        let hzb_image = &self.swapchain_states.hzb_images[frame_index];
        let hzb_set = self.swapchain_states.hzb_sets[frame_index];
        let hzb_build_src_views = &self.swapchain_states.hzb_build_src_views[frame_index];
        let depth_image = &self.swapchain_states.depth_images[frame_index];
        let depth_view = &self.swapchain_states.depth_views[frame_index];

        let depth_info = [vk::DescriptorImageInfo::default()
            .image_view(depth_view.view)
            .sampler(self.swapchain_states.hzb_sampler)
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];
        self.core.device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(hzb_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(depth_info.len() as u32)
                .image_info(&depth_info)],
            &[],
        );

        record_cmd_buffer(&self.core.device, build_hzb, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::BuildHzb, || {
                // HZB is sampled from the previous frame and rewritten by this frame's reduction passes.
                self.core.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        // Prepare the HZB for writing.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .image(hzb_image.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                level_count: vk::REMAINING_MIP_LEVELS,
                                ..COLOR_2D_SUBRESOURCE_RANGE
                            })
                            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::GENERAL),
                        // Prepare the depth buffer for sampling @ first HZB reduction.
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            )
                            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(depth_image.image)
                            .subresource_range(DEPTH_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL),
                    ]),
                );

                let build_hzb = |level: u32| {
                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines.build_hzb_pipeline_layout,
                        0,
                        &[hzb_set],
                        &[],
                    );

                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines.build_hzb_pipeline,
                    );

                    let w = hzb_base_width.checked_shr(level).unwrap_or(0).max(1).div_ceil(8);
                    let h = hzb_base_height.checked_shr(level).unwrap_or(0).max(1).div_ceil(8);
                    self.core.device.cmd_dispatch_base(cmd, 0, 0, level, w, h, 1);

                    // Keep each mip level coherent as the reduction chain walks down the pyramid.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(hzb_image.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                base_mip_level: level,
                                level_count: 1,
                                ..COLOR_2D_SUBRESOURCE_RANGE
                            })
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );
                };

                // For the first compute, the src view is the depth buffer, which depends on the depth buffer.
                let mips = hzb_build_src_views.len() as u32;
                for level in 0..mips {
                    build_hzb(level);
                }

                // Return the HZB to sampled-read and the depth buffer to attachment-write for the next render frame.
                self.core.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(hzb_image.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                level_count: vk::REMAINING_MIP_LEVELS,
                                ..COLOR_2D_SUBRESOURCE_RANGE
                            })
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            )
                            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                            .image(depth_image.image)
                            .subresource_range(DEPTH_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                    ]),
                );
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(build_hzb)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::BuildHzb.wait_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::BuildHzb.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_occlusion_cull_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        pipeline_semaphore: vk::Semaphore,
    ) {
        let fif = &self.fifs[frame_index];
        let profiler = &fif.profiler;
        let occlusion_cull = fif.cmd_buffers[PipelineStage::OcclusionCull as usize];
        let hzb_set = self.swapchain_states.hzb_sets[frame_index];
        let frame_set = fif.frame_set;
        let frame_global_buffer = &fif.frame_global_buffer;
        let indirect_cmd_buffer = &fif.indirect_cmd_buffer;

        record_cmd_buffer(&self.core.device, occlusion_cull, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::OcclusionCull, || {
                // Reuse the indirect buffer for the late list only after early draw has consumed it.
                self.core.device.cmd_fill_buffer(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    std::mem::size_of::<u32>() as u64,
                    0,
                );

                self.core.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.occlusion_cull_pipeline_layout,
                    0,
                    &[hzb_set, frame_set],
                    &[],
                );

                self.core.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines.occlusion_cull_pipeline,
                );

                self.core.device.cmd_dispatch_indirect(
                    cmd,
                    frame_global_buffer.vk_handle(),
                    offset_of!(GpuFrameGlobal, occlusion_dispatch) as u64,
                );
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(occlusion_cull)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::BuildHzb.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::OcclusionCull.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_late_draw_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        image_index: u32,
        pipeline_semaphore: vk::Semaphore,
        debug_draw_enabled: bool,
    ) {
        let fif = &self.fifs[frame_index];
        let profiler = &fif.profiler;
        let late_draw = fif.cmd_buffers[PipelineStage::LateDraw as usize];
        let swapchain_extent = self.swapchain.extent;
        let swapchain_image = self.swapchain.images[image_index as usize];
        let swapchain_view = self.swapchain.views[image_index as usize];
        let depth_view = &self.swapchain_states.depth_views[frame_index];
        let global_set = self.global_set;
        let frame_set = fif.frame_set;
        let overdraw_set = fif.overdraw_set;
        let scene_index_buffer = &fif.scene_index_buffer;
        let indirect_cmd_buffer = &fif.indirect_cmd_buffer;

        record_cmd_buffer(&self.core.device, late_draw, |cmd| {
            profiler.begin(&self.core.device, cmd, PipelineStage::LateDraw, || {
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let render_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: swapchain_extent.width,
                            height: swapchain_extent.height,
                        },
                    })
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                if debug_draw_enabled {
                    self.core.device.cmd_begin_rendering(cmd, &render_info.color_attachments(&[]));
                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.overdraw_render_pipeline_layout,
                        0,
                        &[global_set, overdraw_set],
                        &[],
                    );
                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.overdraw_render_pipeline,
                    );
                } else {
                    self.core.device.cmd_begin_rendering(
                        cmd,
                        &render_info.color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(swapchain_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)]),
                    );

                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.render_pipeline_layout,
                        0,
                        &[global_set, frame_set],
                        &[],
                    );

                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.render_pipeline,
                    );
                }

                self.core.device.cmd_set_viewport(
                    cmd,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: swapchain_extent.width as f32,
                        height: swapchain_extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );

                self.core.device.cmd_set_scissor(
                    cmd,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_extent,
                    }],
                );

                self.core.device.cmd_bind_index_buffer(cmd, scene_index_buffer.vk_handle(), 0, vk::IndexType::UINT32);

                self.core.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    indirect_cmd_buffer.vk_handle(),
                    std::mem::size_of::<GpuIndex>() as u64,
                    indirect_cmd_buffer.vk_handle(),
                    0,
                    indirect_cmd_buffer.len(),
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );

                self.core.device.cmd_end_rendering(cmd);

                if debug_draw_enabled {
                    // In overdraw mode, the swapchain image becomes a storage image for the resolve compute pass.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .new_layout(vk::ImageLayout::GENERAL)]),
                    );

                    self.core.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines.overdraw_resolve_pipeline_layout,
                        0,
                        &[global_set, overdraw_set],
                        &[],
                    );
                    self.core.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines.overdraw_resolve_pipeline,
                    );
                    self.core.device.cmd_dispatch(
                        cmd,
                        swapchain_extent.width.div_ceil(8),
                        swapchain_extent.height.div_ceil(8),
                        1,
                    );

                    // The compute resolve writes the final swapchain image, so transition it back to presentable usage.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .old_layout(vk::ImageLayout::GENERAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)]),
                    );
                }

                if !debug_draw_enabled {
                    // Hand the swapchain image from color attachment output to presentation.
                    self.core.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .image(swapchain_image)
                            .subresource_range(COLOR_2D_SUBRESOURCE_RANGE)
                            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)]),
                    );
                }
            })
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(late_draw)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::OcclusionCull.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::LateDraw.signal_value(frame_timeline_base))
                        .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_frame_end_stage(
        &self,
        frame_index: usize,
        frame_timeline_base: u64,
        pipeline_semaphore: vk::Semaphore,
        render_finished: vk::Semaphore,
        debug_draw_enabled: bool,
    ) {
        let frame_end = self.fifs[frame_index].cmd_buffers[PipelineStage::FrameEnd as usize];

        record_cmd_buffer(&self.core.device, frame_end, |_cmd| {
            // FrameEnd is intentionally empty; it only preserves the stage accounting / timeline structure.
        });

        self.core
            .device
            .queue_submit2(
                *self.core.graphics_queue.lock().unwrap(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default().command_buffer(frame_end)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(pipeline_semaphore)
                        .value(PipelineStage::LateDraw.signal_value(frame_timeline_base))
                        .stage_mask(match debug_draw_enabled {
                            true => vk::PipelineStageFlags2::TOP_OF_PIPE,
                            false => vk::PipelineStageFlags2::COMPUTE_SHADER,
                        })])
                    .signal_semaphore_infos(&[
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(pipeline_semaphore)
                            .value(PipelineStage::FrameEnd.signal_value(frame_timeline_base))
                            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(render_finished)
                            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
                    ])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    unsafe fn record_and_submit_pipeline_stages(
        &mut self,
        frame_index: usize,
        frame_timeline_base: u64,
        world: World,
    ) -> (u32, vk::Semaphore) {
        let pipeline_semaphore = self.fifs[frame_index].timeline;
        let image_acquired = self.swapchain_states.image_acquired_semaphores[frame_index];

        let (image_index, _) = self
            .swapchain
            .swapchain_device
            .acquire_next_image(self.swapchain.swapchain, u64::MAX, image_acquired, vk::Fence::null())
            .unwrap();
        let render_finished = self.swapchain_states.render_finished[image_index as usize];
        let debug_draw_enabled = self.overdraw_enabled || self.overshade_enabled;

        self.read_and_accumulate_frame_profile(frame_index);

        // Object-array-relative dispatch order consumed by FrustumCull. Keeping this separate from object_data allows
        // general CPU-side sorting, though the shader forwards frustum-passing meshlets with atomics, so downstream
        // order is not guaranteed, just usually preserved enough to matter for performance.
        let mut object_dispatch = Vec::new();

        // DataUpload stage. Builds CPU frame data, records transfer/reset commands, and serializes staging upload work.
        let visibility_resource_waits = self.record_and_submit_data_upload_stage(
            frame_index,
            frame_timeline_base,
            pipeline_semaphore,
            &world,
            &mut object_dispatch,
        );

        // Adopt the target snapshot only after its uploads have been submitted.
        // The remaining stages consume those uploads through the FIF timeline.
        self.fifs[frame_index].world = world;

        // FrustumCull stage. Records per-object compute dispatches and waits for upload plus visibility resources.
        self.record_and_submit_frustum_cull_stage(
            frame_index,
            frame_timeline_base,
            pipeline_semaphore,
            object_dispatch,
            visibility_resource_waits,
        );

        // EarlyDraw stage. Records the speculative visible-last-frame draw and waits on swapchain acquisition.
        self.record_and_submit_early_draw_stage(
            frame_index,
            frame_timeline_base,
            image_index,
            pipeline_semaphore,
            image_acquired,
            debug_draw_enabled,
        );

        // BuildHzb stage. Reduces the early depth buffer into the per-FIF HZB image.
        self.record_and_submit_build_hzb_stage(frame_index, frame_timeline_base, pipeline_semaphore);

        // OcclusionCull stage. Rebuilds the late indirect list from frustum candidates and the freshly built HZB.
        self.record_and_submit_occlusion_cull_stage(frame_index, frame_timeline_base, pipeline_semaphore);

        // LateDraw stage. Records newly visible draws and resolves/debug-transitions the swapchain image.
        self.record_and_submit_late_draw_stage(
            frame_index,
            frame_timeline_base,
            image_index,
            pipeline_semaphore,
            debug_draw_enabled,
        );

        // FrameEnd stage. Submits an empty command buffer to publish completion and presentation signals.
        self.record_and_submit_frame_end_stage(
            frame_index,
            frame_timeline_base,
            pipeline_semaphore,
            render_finished,
            debug_draw_enabled,
        );

        (image_index, render_finished)
    }

    pub fn render(&mut self, _timestamp: f32) {
        self.frame += 1;

        // Mesh residency remains shared; each idle FIF reconciles its own scene.
        self.promote_completed_gpu_meshes();
        self.upload_missing_gpu_meshes();
        self.rebuild_swapchain_states_if_dirty();

        let (frame_index, frame_timeline_base) = self.reserve_available_frame_slot();

        unsafe {
            let world = self.reconcile_fif(frame_index);
            let (image_index, render_finished) =
                self.record_and_submit_pipeline_stages(frame_index, frame_timeline_base, world);

            // Present.
            self.swapchain
                .swapchain_device
                .queue_present(
                    *self.core.present_queue.lock().unwrap(),
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[render_finished])
                        .swapchains(&[self.swapchain.swapchain])
                        .image_indices(&[image_index]),
                )
                .unwrap();
        }
    }

    // Called only after this FIF's previous submission has completed.
    unsafe fn reconcile_fif(&mut self, frame_index: usize) -> World {
        let (_world_diff, world) = WorldDiff::between_resident(&self.fifs[frame_index].world, &self.world, |mesh_id| {
            self.gpu_meshes.contains_key(&mesh_id)
        });

        // Incremental uploads are still TODO, but this is the sole boundary at
        // which a canonical object is rejected for lacking a resident mesh.
        // Ensure capacity before planning updates from the local world to the target world.
        let index_count =
            world.meshes.iter().map(|mesh_id| self.gpu_meshes[mesh_id].index_buffer.len()).sum::<u32>().max(1);
        let object_count = (world.objects.len() as u32).max(1);
        let meshlet_count = world
            .objects
            .values()
            .map(|object| self.meshes[&object.mesh].lods.iter().map(|lod| lod.len() as u32).max().unwrap_or(0))
            .sum::<u32>()
            .max(1);

        let fif = &mut self.fifs[frame_index];
        let allocator = &self.core.allocator;
        let storage_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        // Only the idle FIF's undersized buffers are replaced. Invalidate its snapshot
        // so persistent contents are repopulated before the next draw.
        // Count replacement buffer bytes and time allocation only (not destruction or uploads).
        let mut allocated_bytes = 0u64;
        let mut allocation_time = Duration::ZERO;
        if fif.scene_index_buffer.len() < index_count {
            fif.scene_index_buffer.take().destroy(allocator);
            let start = Instant::now();
            fif.scene_index_buffer = Buffer::<[GpuIndex]>::new(
                allocator,
                index_count,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );
            allocation_time += start.elapsed();
            allocated_bytes += fif.scene_index_buffer.size() as u64;
            fif.world = World::default();
        }
        if fif.object_instance_buffer.len() < object_count {
            fif.object_instance_buffer.take().destroy(allocator);
            let start = Instant::now();
            fif.object_instance_buffer = Buffer::<[GpuObjectInstance]>::new(
                allocator,
                object_count,
                storage_usage,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );
            allocation_time += start.elapsed();
            allocated_bytes += fif.object_instance_buffer.size() as u64;
            fif.world = World::default();
        }
        if fif.indirect_cmd_buffer.len() < meshlet_count {
            fif.indirect_cmd_buffer.take().destroy(allocator);
            let start = Instant::now();
            fif.indirect_cmd_buffer = Buffer::<GpuDrawCommandBuffer>::new_trailing(
                allocator,
                meshlet_count,
                storage_usage | vk::BufferUsageFlags::INDIRECT_BUFFER,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );
            allocation_time += start.elapsed();
            allocated_bytes += fif.indirect_cmd_buffer.size() as u64;
            fif.world = World::default();
        }
        if fif.frustum_passing_meshlet_buffer.len() < meshlet_count {
            fif.frustum_passing_meshlet_buffer.take().destroy(allocator);
            let start = Instant::now();
            fif.frustum_passing_meshlet_buffer = Buffer::<GpuFrustumPassingMeshletBuffer>::new_trailing(
                allocator,
                meshlet_count,
                storage_usage,
                vk_mem::MemoryUsage::AutoPreferDevice,
            );
            allocation_time += start.elapsed();
            allocated_bytes += fif.frustum_passing_meshlet_buffer.size() as u64;
            fif.world = World::default();
        }
        if allocated_bytes != 0 {
            println!(
                "FIF {frame_index} resized buffers: {:.2} MiB allocated in {:.3} ms (CPU)",
                allocated_bytes as f64 / MiB as f64,
                allocation_time.as_secs_f64() * 1000.0,
            );
        }

        // TODO: Use the resident world diff to plan incremental updates. Clearing
        // the local world on growth will naturally request full repopulation.
        // For now, rebuild the entire layout and let DataUpload copy every mesh.
        fif.scene_index_offsets.clear();
        let mut offset = 0;
        for mesh_id in &world.meshes {
            fif.scene_index_offsets.insert(*mesh_id, offset);
            offset += self.gpu_meshes[mesh_id].index_buffer.len();
        }

        world
    }

    pub fn create_object(
        &mut self,
        mesh: MeshHandle,
        position: Vec3,
        scale: f32,
        orientation: Quat,
    ) -> Option<ObjectHandle> {
        if !self.meshes.contains_key(&mesh) {
            return None;
        }
        let handle = self.resource_counter.next().unwrap();
        self.world.objects.insert(handle, Object { mesh, position, scale, orientation });
        self.world.meshes.insert(mesh);
        Some(handle)
    }

    pub fn load_mesh(&mut self, filename: impl AsRef<Path>) -> Option<MeshHandle> {
        let mesh = load_mesh(self.cwd.join(filename))?;
        let handle = self.resource_counter.next().unwrap();
        self.meshes.insert(handle, Arc::new(mesh));
        return Some(handle);
    }
}
