// This project requires so much unsafe, its annoying otherwise.
#![allow(unsafe_op_in_unsafe_fn)]

extern crate ash;
extern crate ash_window;
extern crate glam;
extern crate itertools;
extern crate png;
extern crate tobj;
extern crate vk_mem;
extern crate winit;

mod buffer;
mod cache;
mod cache2;
mod core;
mod glsl_types;
mod image;
mod mesh;
mod pipelines;
mod profiling;
mod renderer;
mod rw_queue;
mod staging;
mod swapchain;
mod util;
mod vk_helpers;
mod world;

use crate::renderer::*;
use glam::*;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{DeviceEvents, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

fn main() {
    // Create window.
    let (viewport_w, viewport_h) = (1024_u32, 1024_u32);
    let mut event_loop = EventLoop::new().expect("Could not create window event loop.");
    event_loop.listen_device_events(DeviceEvents::Always);
    #[allow(deprecated)]
    let window = event_loop
        .create_window(
            Window::default_attributes()
                .with_resizable(false)
                .with_inner_size(PhysicalSize::new(viewport_w, viewport_h)),
        )
        .expect("Could not create window.");
    window.set_cursor_visible(false);
    let center = PhysicalPosition::new(viewport_w as f64 * 0.5, viewport_h as f64 * 0.5);
    let _ =
        window.set_cursor_grab(CursorGrabMode::Locked).or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
    let _ = window.set_cursor_position(center);

    let cwd = std::env::current_dir().unwrap();
    let mut renderer = Renderer::new(cwd, viewport_w, viewport_h, &window);
    let viking_room = renderer.load_mesh("resources/models/viking_room.obj").unwrap();
    let sphere = renderer.load_mesh("resources/models/sphere.obj").unwrap();
    let bunny = renderer.load_mesh("resources/models/bunny2.obj").unwrap();
    let _obj0 = renderer
        .create_object(
            viking_room,
            Vec3::new(0.0, 0.5, 0.0),
            1.0,
            Quat::from_euler(EulerRot::XYZ, std::f32::consts::FRAC_PI_2, 0.0, 0.0),
        )
        .unwrap();
    let _obj1 = renderer
        .create_object(sphere, Vec3::new(0.0, 0.0, 0.0), 0.05, Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, 0.0))
        .unwrap();
    let bunny_spacing = 0.25_f32;
    let bunny_scale = 1.4_f32;
    const BUNNY_SIDE: usize = 16;
    let bunny_offset = Vec3::new(
        1.5 - ((BUNNY_SIDE - 1) as f32) * 0.5 * bunny_spacing,
        -0.25 - ((BUNNY_SIDE - 1) as f32) * 0.5 * bunny_spacing,
        -4.0 - ((BUNNY_SIDE - 1) as f32) * 0.5 * bunny_spacing,
    );
    // Deterministic per-instance jitter keeps the scene stable across runs while breaking up uniform silhouettes.
    let next_unit = |seed: &mut u32| -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*seed >> 8) as f32) / (((u32::MAX >> 8) as f32).max(1.0))
    };
    for z in 0..BUNNY_SIDE {
        for y in 0..BUNNY_SIDE {
            for x in 0..BUNNY_SIDE {
                let position = bunny_offset
                    + Vec3::new(x as f32 * bunny_spacing, y as f32 * bunny_spacing, z as f32 * bunny_spacing);
                let mut seed = (x as u32).wrapping_mul(73856093)
                    ^ (y as u32).wrapping_mul(19349663)
                    ^ (z as u32).wrapping_mul(83492791)
                    ^ 0xB16B_00B5;
                let yaw = next_unit(&mut seed) * std::f32::consts::TAU;
                let pitch = (next_unit(&mut seed) - 0.5) * 0.35;
                let roll = (next_unit(&mut seed) - 0.5) * 0.35;
                let _ = renderer
                    .create_object(
                        bunny,
                        position,
                        bunny_scale,
                        Quat::from_euler(EulerRot::XYZ, std::f32::consts::PI + pitch, yaw, roll),
                    )
                    .unwrap();
            }
        }
    }

    // "Gameloop"
    //let mut timestamp = 0_u64;
    let mut time = 0_f32;
    let dt = 0.016666_f32;
    // Misc.
    let mut w_down = false;
    let mut a_down = false;
    let mut s_down = false;
    let mut d_down = false;
    let mut bunny_count = 0;
    loop {
        // Input.
        let mut exit = false;
        let mut mouse_dx = 0.0_f64;
        let mut mouse_dy = 0.0_f64;
        use winit::platform::pump_events::EventLoopExtPumpEvents;
        #[allow(deprecated)]
        let _status = event_loop.pump_events(Some(std::time::Duration::ZERO), |event, _| {
            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => exit = true,
                Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                    mouse_dx += delta.0;
                    mouse_dy += delta.1;
                }

                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    physical_key: PhysicalKey::Code(key), state, repeat: false, ..
                                },
                            ..
                        },
                    ..
                } => {
                    // Skip repeats.
                    let var = match key {
                        KeyCode::KeyW => &mut w_down,
                        KeyCode::KeyA => &mut a_down,
                        KeyCode::KeyS => &mut s_down,
                        KeyCode::KeyD => &mut d_down,
                        KeyCode::KeyO if state == ElementState::Pressed => {
                            renderer.overdraw_enabled = !renderer.overdraw_enabled;
                            if renderer.overdraw_enabled {
                                renderer.overshade_enabled = false;
                            }
                            return;
                        }
                        KeyCode::KeyP if state == ElementState::Pressed => {
                            renderer.overshade_enabled = !renderer.overshade_enabled;
                            if renderer.overshade_enabled {
                                renderer.overdraw_enabled = false;
                            }
                            return;
                        }
                        KeyCode::KeyZ if state == ElementState::Pressed => {
                            let i = bunny_count;
                            bunny_count += 1;
                            renderer
                                .create_object(
                                    bunny,
                                    Vec3::new(1.0 + 0.55 * i as f32, 0.4, 0.0),
                                    bunny_scale,
                                    Quat::from_euler(EulerRot::XYZ, std::f32::consts::PI, 0.0, 0.0),
                                )
                                .unwrap();
                            return;
                        }
                        _ => return,
                    };

                    match state {
                        ElementState::Pressed => *var = true,
                        ElementState::Released => *var = false,
                    }
                }

                // Unhandled.
                _ => {}
            }
        });

        if mouse_dx != 0.0 || mouse_dy != 0.0 {
            renderer.cam_rot[0] += mouse_dx as f32 * 0.0025;
            renderer.cam_rot[1] += mouse_dy as f32 * 0.0025;
            renderer.cam_rot[1] =
                renderer.cam_rot[1].clamp(-std::f32::consts::FRAC_PI_2 + 0.001, std::f32::consts::FRAC_PI_2 - 0.001);
        }

        if exit {
            break;
        }

        // Update.

        // Forward.
        if s_down && !w_down {
            renderer.cam_pos.z += dt * renderer.cam_rot[0].cos();
            renderer.cam_pos.x -= dt * renderer.cam_rot[0].sin();
        }
        // Backward.
        if !s_down && w_down {
            renderer.cam_pos.z -= dt * renderer.cam_rot[0].cos();
            renderer.cam_pos.x += dt * renderer.cam_rot[0].sin();
        }

        // Strafe left.
        if a_down && !d_down {
            renderer.cam_pos.x -= dt * renderer.cam_rot[0].cos();
            renderer.cam_pos.z -= dt * renderer.cam_rot[0].sin();
        }
        // Strafe right.
        if !a_down && d_down {
            renderer.cam_pos.x += dt * renderer.cam_rot[0].cos();
            renderer.cam_pos.z += dt * renderer.cam_rot[0].sin();
        }

        //cam_vr = cam_vr.clamp(-FRAC_PI_2, FRAC_PI_2);

        renderer.render(time);
        let _ = window.set_cursor_position(center);

        //timestamp += 16666;
        time += 0.016666 * 0.1;
        //panic!();
    }

    std::process::abort();
}
