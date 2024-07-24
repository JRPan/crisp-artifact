#ifndef ANV_PUBLIC_H
#define ANV_PUBLIC_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <assert.h>
#include <stdint.h>


struct anv_bo {
   uint32_t gem_handle;

   uint32_t refcount;

   /* Index into the current validation list.  This is used by the
    * validation list building alrogithm to track which buffers are already
    * in the validation list so that we can ensure uniqueness.
    */
   uint32_t index;

   /* Index for use with util_sparse_array_free_list */
   uint32_t free_index;

   /* Last known offset.  This value is provided by the kernel when we
    * execbuf and is used as the presumed offset for the next bunch of
    * relocations.
    */
   uint64_t offset;

   /** Size of the buffer not including implicit aux */
   uint64_t size;

   /* Map for internally mapped BOs.
    *
    * If ANV_BO_WRAPPER is set in flags, map points to the wrapped BO.
    */
   void *map;

   /** Size of the implicit CCS range at the end of the buffer
    *
    * On Gen12, CCS data is always a direct 1/256 scale-down.  A single 64K
    * page of main surface data maps to a 256B chunk of CCS data and that
    * mapping is provided on TGL-LP by the AUX table which maps virtual memory
    * addresses in the main surface to virtual memory addresses for CCS data.
    *
    * Because we can't change these maps around easily and because Vulkan
    * allows two VkImages to be bound to overlapping memory regions (as long
    * as the app is careful), it's not feasible to make this mapping part of
    * the image.  (On Gen11 and earlier, the mapping was provided via
    * RENDER_SURFACE_STATE so each image had its own main -> CCS mapping.)
    * Instead, we attach the CCS data directly to the buffer object and setup
    * the AUX table mapping at BO creation time.
    *
    * This field is for internal tracking use by the BO allocator only and
    * should not be touched by other parts of the code.  If something wants to
    * know if a BO has implicit CCS data, it should instead look at the
    * has_implicit_ccs boolean below.
    *
    * This data is not included in maps of this buffer.
    */
   uint32_t _ccs_size;

   /** Flags to pass to the kernel through drm_i915_exec_object2::flags */
   uint32_t flags;

   /** True if this BO may be shared with other processes */
   bool is_external:1;

   /** True if this BO is a wrapper
    *
    * When set to true, none of the fields in this BO are meaningful except
    * for anv_bo::is_wrapper and anv_bo::map which points to the actual BO.
    * See also anv_bo_unwrap().  Wrapper BOs are not allowed when use_softpin
    * is set in the physical device.
    */
   bool is_wrapper:1;

   /** See also ANV_BO_ALLOC_FIXED_ADDRESS */
   bool has_fixed_address:1;

   /** True if this BO wraps a host pointer */
   bool from_host_ptr:1;

   /** See also ANV_BO_ALLOC_CLIENT_VISIBLE_ADDRESS */
   bool has_client_visible_address:1;

   /** True if this BO has implicit CCS data attached to it */
   bool has_implicit_ccs:1;
};

struct anv_address {
   struct anv_bo *bo;
   uint64_t offset;
};

static inline void *
anv_address_map(struct anv_address addr)
{
   if (addr.bo == NULL)
      return NULL;

   if (addr.bo->map == NULL)
      return NULL;

   return addr.bo->map + addr.offset;
}

struct GENERAL_SBT {
    uint64_t General;
};

struct TRIANGLES_SBT {
   uint64_t ClosestHit;
   uint64_t AnyHit;
};

struct PROCEDURAL_SBT {
   uint64_t ClosestHit;
   uint64_t Intersection;
};

#endif /* ANV_PUBLIC_H */