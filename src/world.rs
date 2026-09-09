use crate::renderer::{MeshHandle, ObjectHandle};
use glam::{Quat, Vec3};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Object {
    pub mesh: MeshHandle,
    pub position: Vec3,
    pub scale: f32,
    pub orientation: Quat,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct World {
    pub objects: BTreeMap<ObjectHandle, Object>,
    pub meshes: BTreeSet<MeshHandle>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct WorldDiff {
    pub added_objects: BTreeSet<ObjectHandle>,
    pub removed_objects: BTreeSet<ObjectHandle>,
    pub changed_objects: BTreeSet<ObjectHandle>,
    pub added_meshes: BTreeSet<MeshHandle>,
    pub removed_meshes: BTreeSet<MeshHandle>,
}

impl WorldDiff {
    pub(crate) fn between(previous: &World, next: &World) -> Self {
        let previous_objects: BTreeSet<_> = previous.objects.keys().copied().collect();
        let next_objects: BTreeSet<_> = next.objects.keys().copied().collect();

        let changed_objects = previous_objects
            .intersection(&next_objects)
            .copied()
            .filter(|handle| previous.objects.get(handle) != next.objects.get(handle))
            .collect();

        Self {
            added_objects: next_objects.difference(&previous_objects).copied().collect(),
            removed_objects: previous_objects.difference(&next_objects).copied().collect(),
            changed_objects,
            added_meshes: next.meshes.difference(&previous.meshes).copied().collect(),
            removed_meshes: previous.meshes.difference(&next.meshes).copied().collect(),
        }
    }

    // A FIF may only encode objects whose mesh payload is resident. Keep that
    // decision next to the snapshot diff rather than teaching every consumer
    // of World about missing GPU meshes.
    pub(crate) fn between_resident(
        previous: &World,
        canonical: &World,
        mesh_is_resident: impl Fn(MeshHandle) -> bool,
    ) -> (Self, World) {
        let target = World {
            objects: canonical
                .objects
                .iter()
                .filter(|(_, object)| mesh_is_resident(object.mesh))
                .map(|(&handle, &object)| (handle, object))
                .collect(),
            meshes: canonical.meshes.iter().filter(|&&mesh| mesh_is_resident(mesh)).copied().collect(),
        };
        (Self::between(previous, &target), target)
    }
}
