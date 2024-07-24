/*
 * Copyright © 2014 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Connor Abbott (cwabbott0@gmail.com)
 *
 */

#include "nir.h"
#include "compiler/shader_enums.h"
#include "util/half_float.h"
#include "vulkan/vulkan_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h> /* for PRIx64 macro */

static void
print_tabs(unsigned num_tabs, FILE *fp)
{
   for (unsigned i = 0; i < num_tabs; i++)
      fprintf(fp, "\t");
}

typedef struct {
   FILE *fp;
   nir_shader *shader;
   /** map from nir_variable -> printable name */
   struct hash_table *ht;

   /** set of names used so far for nir_variables */
   struct set *syms;

   /* an index used to make new non-conflicting names */
   unsigned index;

   /**
    * Optional table of annotations mapping nir object
    * (such as instr or var) to message to print.
    */
   struct hash_table *annotations;
} print_state;

static void
print_annotation(print_state *state, void *obj)
{
   FILE *fp = state->fp;

   if (!state->annotations)
      return;

   struct hash_entry *entry = _mesa_hash_table_search(state->annotations, obj);
   if (!entry)
      return;

   const char *note = entry->data;
   _mesa_hash_table_remove(state->annotations, entry);

   fprintf(fp, "%s\n\n", note);
}

static void
print_register(nir_register *reg, print_state *state)
{
   FILE *fp = state->fp;
   if (reg->name != NULL)
      fprintf(fp, "/* %s */ ", reg->name);
   fprintf(fp, "r%u", reg->index);
}

static const char *sizes[] = { "error", "vec1", "vec2", "vec3", "vec4",
                               "vec5", "error", "error", "vec8",
                               "error", "error", "error", "error",
                               "error", "error", "error", "vec16"};

static void
print_register_decl(nir_register *reg, print_state *state)
{
   FILE *fp = state->fp;
   fprintf(fp, "decl_reg %s %u ", sizes[reg->num_components], reg->bit_size);
   print_register(reg, state);
   if (reg->num_array_elems != 0)
      fprintf(fp, "[%u]", reg->num_array_elems);
   fprintf(fp, "\n");
}

static void
print_ssa_def(nir_ssa_def *def, print_state *state)
{
   FILE *fp = state->fp;
   if (def->name != NULL)
      fprintf(fp, "/* %s */ ", def->name);
   fprintf(fp, "%s %u ssa_%u", sizes[def->num_components], def->bit_size,
           def->index);
}

static void
print_ssa_use(nir_ssa_def *def, print_state *state)
{
   FILE *fp = state->fp;
   if (def->name != NULL)
      fprintf(fp, "/* %s */ ", def->name);
   fprintf(fp, "ssa_%u", def->index);
}

static void print_src(const nir_src *src, print_state *state);

static void
print_reg_src(const nir_reg_src *src, print_state *state)
{
   FILE *fp = state->fp;
   print_register(src->reg, state);
   if (src->reg->num_array_elems != 0) {
      fprintf(fp, "[%u", src->base_offset);
      if (src->indirect != NULL) {
         fprintf(fp, " + ");
         print_src(src->indirect, state);
      }
      fprintf(fp, "]");
   }
}

static void
print_reg_dest(nir_reg_dest *dest, print_state *state)
{
   FILE *fp = state->fp;
   print_register(dest->reg, state);
   if (dest->reg->num_array_elems != 0) {
      fprintf(fp, "[%u", dest->base_offset);
      if (dest->indirect != NULL) {
         fprintf(fp, " + ");
         print_src(dest->indirect, state);
      }
      fprintf(fp, "]");
   }
}

static void
print_src(const nir_src *src, print_state *state)
{
   if (src->is_ssa)
      print_ssa_use(src->ssa, state);
   else
      print_reg_src(&src->reg, state);
}

static void
print_dest(nir_dest *dest, print_state *state)
{
   if (dest->is_ssa)
      print_ssa_def(&dest->ssa, state);
   else
      print_reg_dest(&dest->reg, state);
}

static const char *
comp_mask_string(unsigned num_components)
{
   return (num_components > 4) ? "abcdefghijklmnop" : "xyzw";
}

static void
print_alu_src(nir_alu_instr *instr, unsigned src, print_state *state)
{
   FILE *fp = state->fp;

   if (instr->src[src].negate)
      fprintf(fp, "-");
   if (instr->src[src].abs)
      fprintf(fp, "abs(");

   print_src(&instr->src[src].src, state);

   bool print_swizzle = false;
   nir_component_mask_t used_channels = 0;

   for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++) {
      if (!nir_alu_instr_channel_used(instr, src, i))
         continue;

      used_channels++;

      if (instr->src[src].swizzle[i] != i) {
         print_swizzle = true;
         break;
      }
   }

   unsigned live_channels = nir_src_num_components(instr->src[src].src);

   if (print_swizzle || used_channels != live_channels) {
      fprintf(fp, ".");
      for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++) {
         if (!nir_alu_instr_channel_used(instr, src, i))
            continue;

         fprintf(fp, "%c", comp_mask_string(live_channels)[instr->src[src].swizzle[i]]);
      }
   }

   if (instr->src[src].abs)
      fprintf(fp, ")");
}

static void
print_alu_dest(nir_alu_dest *dest, print_state *state)
{
   FILE *fp = state->fp;
   /* we're going to print the saturate modifier later, after the opcode */

   print_dest(&dest->dest, state);

   if (!dest->dest.is_ssa &&
       dest->write_mask != (1 << dest->dest.reg.reg->num_components) - 1) {
      unsigned live_channels = dest->dest.reg.reg->num_components;
      fprintf(fp, ".");
      for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++)
         if ((dest->write_mask >> i) & 1)
            fprintf(fp, "%c", comp_mask_string(live_channels)[i]);
   }
}

static void
print_alu_instr(nir_alu_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   print_alu_dest(&instr->dest, state);

   fprintf(fp, " = %s", nir_op_infos[instr->op].name);
   if (instr->exact)
      fprintf(fp, "!");
   if (instr->dest.saturate)
      fprintf(fp, ".sat");
   if (instr->no_signed_wrap)
      fprintf(fp, ".nsw");
   if (instr->no_unsigned_wrap)
      fprintf(fp, ".nuw");
   fprintf(fp, " ");

   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      print_alu_src(instr, i, state);
   }
}

static const char *
get_var_name(nir_variable *var, print_state *state)
{
   if (state->ht == NULL)
      return var->name ? var->name : "unnamed";

   assert(state->syms);

   struct hash_entry *entry = _mesa_hash_table_search(state->ht, var);
   if (entry)
      return entry->data;

   char *name;
   if (var->name == NULL) {
      name = ralloc_asprintf(state->syms, "@%u", state->index++);
   } else {
      struct set_entry *set_entry = _mesa_set_search(state->syms, var->name);
      if (set_entry != NULL) {
         /* we have a collision with another name, append an @ + a unique
          * index */
         name = ralloc_asprintf(state->syms, "%s@%u", var->name,
                                state->index++);
      } else {
         /* Mark this one as seen */
         _mesa_set_add(state->syms, var->name);
         name = var->name;
      }
   }

   _mesa_hash_table_insert(state->ht, var, name);

   return name;
}

static const char *
get_constant_sampler_addressing_mode(enum cl_sampler_addressing_mode mode)
{
   switch (mode) {
   case SAMPLER_ADDRESSING_MODE_NONE: return "none";
   case SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE: return "clamp_to_edge";
   case SAMPLER_ADDRESSING_MODE_CLAMP: return "clamp";
   case SAMPLER_ADDRESSING_MODE_REPEAT: return "repeat";
   case SAMPLER_ADDRESSING_MODE_REPEAT_MIRRORED: return "repeat_mirrored";
   default: unreachable("Invalid addressing mode");
   }
}

static const char *
get_constant_sampler_filter_mode(enum cl_sampler_filter_mode mode)
{
   switch (mode) {
   case SAMPLER_FILTER_MODE_NEAREST: return "nearest";
   case SAMPLER_FILTER_MODE_LINEAR: return "linear";
   default: unreachable("Invalid filter mode");
   }
}

static void
print_constant(nir_constant *c, const struct glsl_type *type, print_state *state)
{
   FILE *fp = state->fp;
   const unsigned rows = glsl_get_vector_elements(type);
   const unsigned cols = glsl_get_matrix_columns(type);
   unsigned i;

   switch (glsl_get_base_type(type)) {
   case GLSL_TYPE_BOOL:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < rows; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "%s", c->values[i].b ? "true" : "false");
      }
      break;

   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < rows; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "0x%02x", c->values[i].u8);
      }
      break;

   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < rows; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "0x%04x", c->values[i].u16);
      }
      break;

   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < rows; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "0x%08x", c->values[i].u32);
      }
      break;

   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_DOUBLE:
      if (cols > 1) {
         for (i = 0; i < cols; i++) {
            if (i > 0) fprintf(fp, ", ");
            print_constant(c->elements[i], glsl_get_column_type(type), state);
         }
      } else {
         switch (glsl_get_base_type(type)) {
         case GLSL_TYPE_FLOAT16:
            for (i = 0; i < rows; i++) {
               if (i > 0) fprintf(fp, ", ");
               fprintf(fp, "%f", _mesa_half_to_float(c->values[i].u16));
            }
            break;

         case GLSL_TYPE_FLOAT:
            for (i = 0; i < rows; i++) {
               if (i > 0) fprintf(fp, ", ");
               fprintf(fp, "%f", c->values[i].f32);
            }
            break;

         case GLSL_TYPE_DOUBLE:
            for (i = 0; i < rows; i++) {
               if (i > 0) fprintf(fp, ", ");
               fprintf(fp, "%f", c->values[i].f64);
            }
            break;

         default:
            unreachable("Cannot get here from the first level switch");
         }
      }
      break;

   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
      /* Only float base types can be matrices. */
      assert(cols == 1);

      for (i = 0; i < cols; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "0x%08" PRIx64, c->values[i].u64);
      }
      break;

   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE:
      for (i = 0; i < c->num_elements; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "{ ");
         print_constant(c->elements[i], glsl_get_struct_field(type, i), state);
         fprintf(fp, " }");
      }
      break;

   case GLSL_TYPE_ARRAY:
      for (i = 0; i < c->num_elements; i++) {
         if (i > 0) fprintf(fp, ", ");
         fprintf(fp, "{ ");
         print_constant(c->elements[i], glsl_get_array_element(type), state);
         fprintf(fp, " }");
      }
      break;

   default:
      unreachable("not reached");
   }
}

static const char *
get_variable_mode_str(nir_variable_mode mode, bool want_local_global_mode)
{
   switch (mode) {
   case nir_var_shader_in:
      return "shader_in";
   case nir_var_shader_out:
      return "shader_out";
   case nir_var_uniform:
      return "uniform";
   case nir_var_mem_ubo:
      return "ubo";
   case nir_var_system_value:
      return "system";
   case nir_var_mem_ssbo:
      return "ssbo";
   case nir_var_mem_shared:
      return "shared";
   case nir_var_mem_global:
      return "global";
   case nir_var_mem_push_const:
      return "push_const";
   case nir_var_mem_constant:
      return "constant";
   case nir_var_shader_temp:
      return want_local_global_mode ? "shader_temp" : "";
   case nir_var_function_temp:
      return want_local_global_mode ? "function_temp" : "";
   case nir_var_shader_call_data:
      return "shader_call_data";
   case nir_var_ray_hit_attrib:
      return "ray_hit_attrib";
   default:
      return "";
   }
}

static void
print_var_decl(nir_variable *var, print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, "decl_var ");

   const char *const cent = (var->data.centroid) ? "centroid " : "";
   const char *const samp = (var->data.sample) ? "sample " : "";
   const char *const patch = (var->data.patch) ? "patch " : "";
   const char *const inv = (var->data.invariant) ? "invariant " : "";
   const char *const per_view = (var->data.per_view) ? "per_view " : "";
   fprintf(fp, "%s%s%s%s%s%s %s ",
           cent, samp, patch, inv, per_view,
           get_variable_mode_str(var->data.mode, false),
           glsl_interp_mode_name(var->data.interpolation));

   enum gl_access_qualifier access = var->data.access;
   const char *const coher = (access & ACCESS_COHERENT) ? "coherent " : "";
   const char *const volat = (access & ACCESS_VOLATILE) ? "volatile " : "";
   const char *const restr = (access & ACCESS_RESTRICT) ? "restrict " : "";
   const char *const ronly = (access & ACCESS_NON_WRITEABLE) ? "readonly " : "";
   const char *const wonly = (access & ACCESS_NON_READABLE) ? "writeonly " : "";
   const char *const reorder = (access & ACCESS_CAN_REORDER) ? "reorderable " : "";
   fprintf(fp, "%s%s%s%s%s%s", coher, volat, restr, ronly, wonly, reorder);

   if (glsl_get_base_type(glsl_without_array(var->type)) == GLSL_TYPE_IMAGE) {
      fprintf(fp, "%s ", util_format_short_name(var->data.image.format));
   }

   if (var->data.precision) {
      const char *precisions[] = {
         "",
         "highp",
         "mediump",
         "lowp",
      };
      fprintf(fp, "%s ", precisions[var->data.precision]);
   }

   fprintf(fp, "%s %s", glsl_get_type_name(var->type),
           get_var_name(var, state));

   if (var->data.mode == nir_var_shader_in ||
       var->data.mode == nir_var_shader_out ||
       var->data.mode == nir_var_uniform ||
       var->data.mode == nir_var_mem_ubo ||
       var->data.mode == nir_var_mem_ssbo) {
      const char *loc = NULL;
      char buf[4];

      switch (state->shader->info.stage) {
      case MESA_SHADER_VERTEX:
         if (var->data.mode == nir_var_shader_in)
            loc = gl_vert_attrib_name(var->data.location);
         else if (var->data.mode == nir_var_shader_out)
            loc = gl_varying_slot_name_for_stage(var->data.location,
                                                 state->shader->info.stage);
         break;
      case MESA_SHADER_GEOMETRY:
         if ((var->data.mode == nir_var_shader_in) ||
             (var->data.mode == nir_var_shader_out)) {
            loc = gl_varying_slot_name_for_stage(var->data.location,
                                                 state->shader->info.stage);
         }
         break;
      case MESA_SHADER_FRAGMENT:
         if (var->data.mode == nir_var_shader_in) {
            loc = gl_varying_slot_name_for_stage(var->data.location,
                                                 state->shader->info.stage);
         } else if (var->data.mode == nir_var_shader_out) {
            loc = gl_frag_result_name(var->data.location);
         }
         break;
      case MESA_SHADER_TESS_CTRL:
      case MESA_SHADER_TESS_EVAL:
      case MESA_SHADER_COMPUTE:
      case MESA_SHADER_KERNEL:
      default:
         /* TODO */
         break;
      }

      if (!loc) {
         if (var->data.location == ~0) {
            loc = "~0";
         } else {
            snprintf(buf, sizeof(buf), "%u", var->data.location);
            loc = buf;
         }
      }

      /* For shader I/O vars that have been split to components or packed,
       * print the fractional location within the input/output.
       */
      unsigned int num_components =
         glsl_get_components(glsl_without_array(var->type));
      const char *components = NULL;
      char components_local[18] = {'.' /* the rest is 0-filled */};
      switch (var->data.mode) {
      case nir_var_shader_in:
      case nir_var_shader_out:
         if (num_components < 16 && num_components != 0) {
            const char *xyzw = comp_mask_string(num_components);
            for (int i = 0; i < num_components; i++)
               components_local[i + 1] = xyzw[i + var->data.location_frac];

            components = components_local;
         }
         break;
      default:
         break;
      }

      fprintf(fp, " (%s%s, %u, %u)%s", loc,
              components ? components : "",
              var->data.driver_location, var->data.binding,
              var->data.compact ? " compact" : "");
   }

   if (var->constant_initializer) {
      fprintf(fp, " = { ");
      print_constant(var->constant_initializer, var->type, state);
      fprintf(fp, " }");
   }
   if (glsl_type_is_sampler(var->type) && var->data.sampler.is_inline_sampler) {
      fprintf(fp, " = { %s, %s, %s }",
              get_constant_sampler_addressing_mode(var->data.sampler.addressing_mode),
              var->data.sampler.normalized_coordinates ? "true" : "false",
              get_constant_sampler_filter_mode(var->data.sampler.filter_mode));
   }
   if (var->pointer_initializer)
      fprintf(fp, " = &%s", get_var_name(var->pointer_initializer, state));

   fprintf(fp, "\n");
   print_annotation(state, var);
}

static void
print_deref_link(const nir_deref_instr *instr, bool whole_chain, print_state *state)
{
   FILE *fp = state->fp;

   if (instr->deref_type == nir_deref_type_var) {
      fprintf(fp, "%s", get_var_name(instr->var, state));
      return;
   } else if (instr->deref_type == nir_deref_type_cast) {
      fprintf(fp, "(%s *)", glsl_get_type_name(instr->type));
      print_src(&instr->parent, state);
      return;
   }

   assert(instr->parent.is_ssa);
   nir_deref_instr *parent =
      nir_instr_as_deref(instr->parent.ssa->parent_instr);

   /* Is the parent we're going to print a bare cast? */
   const bool is_parent_cast =
      whole_chain && parent->deref_type == nir_deref_type_cast;

   /* If we're not printing the whole chain, the parent we print will be a SSA
    * value that represents a pointer.  The only deref type that naturally
    * gives a pointer is a cast.
    */
   const bool is_parent_pointer =
      !whole_chain || parent->deref_type == nir_deref_type_cast;

   /* Struct derefs have a nice syntax that works on pointers, arrays derefs
    * do not.
    */
   const bool need_deref =
      is_parent_pointer && instr->deref_type != nir_deref_type_struct;

   /* Cast need extra parens and so * dereferences */
   if (is_parent_cast || need_deref)
      fprintf(fp, "(");

   if (need_deref)
      fprintf(fp, "*");

   if (whole_chain) {
      print_deref_link(parent, whole_chain, state);
   } else {
      print_src(&instr->parent, state);
   }

   if (is_parent_cast || need_deref)
      fprintf(fp, ")");

   switch (instr->deref_type) {
   case nir_deref_type_struct:
      fprintf(fp, "%s%s", is_parent_pointer ? "->" : ".",
              glsl_get_struct_elem_name(parent->type, instr->strct.index));
      break;

   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array: {
      if (nir_src_is_const(instr->arr.index)) {
         fprintf(fp, "[%"PRId64"]", nir_src_as_int(instr->arr.index));
      } else {
         fprintf(fp, "[");
         print_src(&instr->arr.index, state);
         fprintf(fp, "]");
      }
      break;
   }

   case nir_deref_type_array_wildcard:
      fprintf(fp, "[*]");
      break;

   default:
      unreachable("Invalid deref instruction type");
   }
}

static void
print_deref_instr(nir_deref_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   print_dest(&instr->dest, state);

   switch (instr->deref_type) {
   case nir_deref_type_var:
      fprintf(fp, " = deref_var ");
      break;
   case nir_deref_type_array:
   case nir_deref_type_array_wildcard:
      fprintf(fp, " = deref_array ");
      break;
   case nir_deref_type_struct:
      fprintf(fp, " = deref_struct ");
      break;
   case nir_deref_type_cast:
      fprintf(fp, " = deref_cast ");
      break;
   case nir_deref_type_ptr_as_array:
      fprintf(fp, " = deref_ptr_as_array ");
      break;
   default:
      unreachable("Invalid deref instruction type");
   }

   /* Only casts naturally return a pointer type */
   if (instr->deref_type != nir_deref_type_cast)
      fprintf(fp, "&");

   print_deref_link(instr, false, state);

   fprintf(fp, " (");
   unsigned modes = instr->modes;
   while (modes) {
      int m = u_bit_scan(&modes);
      fprintf(fp, "%s%s", get_variable_mode_str(1 << m, true),
                          modes ? "|" : "");
   }
   fprintf(fp, " %s) ", glsl_get_type_name(instr->type));

   if (instr->deref_type != nir_deref_type_var &&
       instr->deref_type != nir_deref_type_cast) {
      /* Print the entire chain as a comment */
      fprintf(fp, "/* &");
      print_deref_link(instr, true, state);
      fprintf(fp, " */");
   }

   if (instr->deref_type == nir_deref_type_cast) {
      fprintf(fp, " /* ptr_stride=%u, align_mul=%u, align_offset=%u */",
              instr->cast.ptr_stride,
              instr->cast.align_mul, instr->cast.align_offset);
   }
}

static const char *
vulkan_descriptor_type_name(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER: return "sampler";
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "texture+sampler";
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "texture";
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "image";
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: return "texture-buffer";
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return "image-buffer";
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "UBO";
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "SSBO";
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: return "UBO";
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return "SSBO";
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return "input-att";
   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT: return "inline-UBO";
   case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: return "accel-struct";
   default: return "unknown";
   }
}

static void
print_alu_type(nir_alu_type type, print_state *state)
{
   FILE *fp = state->fp;
   unsigned size = nir_alu_type_get_type_size(type);
   const char *name;

   switch (nir_alu_type_get_base_type(type)) {
   case nir_type_int: name = "int"; break;
   case nir_type_uint: name = "uint"; break;
   case nir_type_bool: name = "bool"; break;
   case nir_type_float: name = "float"; break;
   default: name = "invalid";
   }
   if (size)
      fprintf(fp, "%s%u", name, size);
   else
      fprintf(fp, "%s", name);
}

static void
print_intrinsic_instr(nir_intrinsic_instr *instr, print_state *state)
{
   const nir_intrinsic_info *info = &nir_intrinsic_infos[instr->intrinsic];
   unsigned num_srcs = info->num_srcs;
   FILE *fp = state->fp;

   if (info->has_dest) {
      print_dest(&instr->dest, state);
      fprintf(fp, " = ");
   }

   fprintf(fp, "intrinsic %s (", info->name);

   for (unsigned i = 0; i < num_srcs; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      print_src(&instr->src[i], state);
   }

   fprintf(fp, ") (");

   for (unsigned i = 0; i < info->num_indices; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      fprintf(fp, "%d", instr->const_index[i]);
   }

   fprintf(fp, ")");

   for (unsigned i = 0; i < info->num_indices; i++) {
      unsigned idx = info->indices[i];
      fprintf(fp, " /*");
      switch (idx) {
      case NIR_INTRINSIC_WRITE_MASK: {
         /* special case wrmask to show it as a writemask.. */
         unsigned wrmask = nir_intrinsic_write_mask(instr);
         fprintf(fp, " wrmask=");
         for (unsigned i = 0; i < instr->num_components; i++)
            if ((wrmask >> i) & 1)
               fprintf(fp, "%c", comp_mask_string(instr->num_components)[i]);
         break;
      }

      case NIR_INTRINSIC_REDUCTION_OP: {
         nir_op reduction_op = nir_intrinsic_reduction_op(instr);
         fprintf(fp, " reduction_op=%s", nir_op_infos[reduction_op].name);
         break;
      }

      case NIR_INTRINSIC_IMAGE_DIM: {
         static const char *dim_name[] = {
            [GLSL_SAMPLER_DIM_1D] = "1D",
            [GLSL_SAMPLER_DIM_2D] = "2D",
            [GLSL_SAMPLER_DIM_3D] = "3D",
            [GLSL_SAMPLER_DIM_CUBE] = "Cube",
            [GLSL_SAMPLER_DIM_RECT] = "Rect",
            [GLSL_SAMPLER_DIM_BUF] = "Buf",
            [GLSL_SAMPLER_DIM_MS] = "2D-MSAA",
            [GLSL_SAMPLER_DIM_SUBPASS] = "Subpass",
            [GLSL_SAMPLER_DIM_SUBPASS_MS] = "Subpass-MSAA",
         };
         enum glsl_sampler_dim dim = nir_intrinsic_image_dim(instr);
         assert(dim < ARRAY_SIZE(dim_name) && dim_name[dim]);
         fprintf(fp, " image_dim=%s", dim_name[dim]);
         break;
      }

      case NIR_INTRINSIC_IMAGE_ARRAY: {
         bool array = nir_intrinsic_image_array(instr);
         fprintf(fp, " image_array=%s", array ? "true" : "false");
         break;
      }

      case NIR_INTRINSIC_FORMAT: {
         enum pipe_format format = nir_intrinsic_format(instr);
         fprintf(fp, " format=%s ", util_format_short_name(format));
         break;
      }

      case NIR_INTRINSIC_DESC_TYPE: {
         VkDescriptorType desc_type = nir_intrinsic_desc_type(instr);
         fprintf(fp, " desc_type=%s", vulkan_descriptor_type_name(desc_type));
         break;
      }

      case NIR_INTRINSIC_SRC_TYPE: {
         fprintf(fp, " src_type=");
         print_alu_type(nir_intrinsic_src_type(instr), state);
         break;
      }

      case NIR_INTRINSIC_DEST_TYPE: {
         fprintf(fp, " dest_type=");
         print_alu_type(nir_intrinsic_dest_type(instr), state);
         break;
      }

      case NIR_INTRINSIC_SWIZZLE_MASK: {
         fprintf(fp, " swizzle_mask=");
         unsigned mask = nir_intrinsic_swizzle_mask(instr);
         if (instr->intrinsic == nir_intrinsic_quad_swizzle_amd) {
            for (unsigned i = 0; i < 4; i++)
               fprintf(fp, "%d", (mask >> (i * 2) & 3));
         } else if (instr->intrinsic == nir_intrinsic_masked_swizzle_amd) {
            fprintf(fp, "((id & %d) | %d) ^ %d", mask & 0x1F,
                                                (mask >> 5) & 0x1F,
                                                (mask >> 10) & 0x1F);
         } else {
            fprintf(fp, "%d", mask);
         }
         break;
      }

      case NIR_INTRINSIC_MEMORY_SEMANTICS: {
         nir_memory_semantics semantics = nir_intrinsic_memory_semantics(instr);
         fprintf(fp, " mem_semantics=");
         switch (semantics & (NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE)) {
         case 0:                  fprintf(fp, "NONE");    break;
         case NIR_MEMORY_ACQUIRE: fprintf(fp, "ACQ");     break;
         case NIR_MEMORY_RELEASE: fprintf(fp, "REL");     break;
         default:                 fprintf(fp, "ACQ|REL"); break;
         }
         if (semantics & (NIR_MEMORY_MAKE_AVAILABLE)) fprintf(fp, "|AVAILABLE");
         if (semantics & (NIR_MEMORY_MAKE_VISIBLE))   fprintf(fp, "|VISIBLE");
         break;
      }

      case NIR_INTRINSIC_MEMORY_MODES: {
         fprintf(fp, " mem_modes=");
         unsigned int modes = nir_intrinsic_memory_modes(instr);
         while (modes) {
            nir_variable_mode m = u_bit_scan(&modes);
            fprintf(fp, "%s%s", get_variable_mode_str(1 << m, true), modes ? "|" : "");
         }
         break;
      }

      case NIR_INTRINSIC_EXECUTION_SCOPE:
      case NIR_INTRINSIC_MEMORY_SCOPE: {
         fprintf(fp, " %s=", nir_intrinsic_index_names[idx]);
         nir_scope scope =
            idx == NIR_INTRINSIC_MEMORY_SCOPE ? nir_intrinsic_memory_scope(instr)
                                              : nir_intrinsic_execution_scope(instr);
         switch (scope) {
         case NIR_SCOPE_NONE:         fprintf(fp, "NONE");         break;
         case NIR_SCOPE_DEVICE:       fprintf(fp, "DEVICE");       break;
         case NIR_SCOPE_QUEUE_FAMILY: fprintf(fp, "QUEUE_FAMILY"); break;
         case NIR_SCOPE_WORKGROUP:    fprintf(fp, "WORKGROUP");    break;
         case NIR_SCOPE_SHADER_CALL:  fprintf(fp, "SHADER_CALL");  break;
         case NIR_SCOPE_SUBGROUP:     fprintf(fp, "SUBGROUP");     break;
         case NIR_SCOPE_INVOCATION:   fprintf(fp, "INVOCATION");   break;
         }
         break;
      }

      case NIR_INTRINSIC_IO_SEMANTICS:
         fprintf(fp, " location=%u slots=%u",
                 nir_intrinsic_io_semantics(instr).location,
                 nir_intrinsic_io_semantics(instr).num_slots);
         if (state->shader) {
            if (state->shader->info.stage == MESA_SHADER_FRAGMENT &&
                instr->intrinsic == nir_intrinsic_store_output &&
                nir_intrinsic_io_semantics(instr).dual_source_blend_index) {
               fprintf(fp, " dualsrc=1");
            }
            if (state->shader->info.stage == MESA_SHADER_FRAGMENT &&
                instr->intrinsic == nir_intrinsic_load_output &&
                nir_intrinsic_io_semantics(instr).fb_fetch_output) {
               fprintf(fp, " fbfetch=1");
            }
            if (instr->intrinsic == nir_intrinsic_store_output &&
                nir_intrinsic_io_semantics(instr).per_view) {
               fprintf(fp, " perview=1");
            }
            if (state->shader->info.stage == MESA_SHADER_GEOMETRY &&
                instr->intrinsic == nir_intrinsic_store_output) {
               unsigned gs_streams = nir_intrinsic_io_semantics(instr).gs_streams;
               fprintf(fp, " gs_streams(");
               for (unsigned i = 0; i < 4; i++) {
                  fprintf(fp, "%s%c=%u", i ? " " : "", "xyzw"[i],
                          (gs_streams >> (i * 2)) & 0x3);
               }
               fprintf(fp, ")");
            }
            if (state->shader->info.stage == MESA_SHADER_FRAGMENT &&
                nir_intrinsic_io_semantics(instr).medium_precision) {
               fprintf(fp, " mediump");
            }
         }
         break;

      case NIR_INTRINSIC_ROUNDING_MODE: {
         fprintf(fp, " rounding_mode=");
         switch (nir_intrinsic_rounding_mode(instr)) {
         case nir_rounding_mode_undef: fprintf(fp, "undef");   break;
         case nir_rounding_mode_rtne:  fprintf(fp, "rtne");    break;
         case nir_rounding_mode_ru:    fprintf(fp, "ru");      break;
         case nir_rounding_mode_rd:    fprintf(fp, "rd");      break;
         case nir_rounding_mode_rtz:   fprintf(fp, "rtz");     break;
         default:                      fprintf(fp, "unkown");  break;
         }
         break;
      }

      default: {
         unsigned off = info->index_map[idx] - 1;
         fprintf(fp, " %s=%d", nir_intrinsic_index_names[idx], instr->const_index[off]);
         break;
      }
      }
      fprintf(fp, " */");
   }

   if (!state->shader)
      return;

   nir_variable_mode var_mode;
   switch (instr->intrinsic) {
   case nir_intrinsic_load_uniform:
      var_mode = nir_var_uniform;
      break;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_per_vertex_input:
      var_mode = nir_var_shader_in;
      break;
   case nir_intrinsic_load_output:
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
      var_mode = nir_var_shader_out;
      break;
   default:
      return;
   }

   nir_foreach_variable_with_modes(var, state->shader, var_mode) {
      if ((var->data.driver_location == nir_intrinsic_base(instr)) &&
          (instr->intrinsic == nir_intrinsic_load_uniform ||
           (nir_intrinsic_component(instr) >= var->data.location_frac  &&
            nir_intrinsic_component(instr) <
            (var->data.location_frac + glsl_get_components(var->type)))) &&
           var->name) {
         fprintf(fp, "\t/* %s */", var->name);
         break;
      }
   }
}

static void
print_tex_instr(nir_tex_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   print_dest(&instr->dest, state);

   fprintf(fp, " = (");
   print_alu_type(instr->dest_type, state);
   fprintf(fp, ")");

   switch (instr->op) {
   case nir_texop_tex:
      fprintf(fp, "tex ");
      break;
   case nir_texop_txb:
      fprintf(fp, "txb ");
      break;
   case nir_texop_txl:
      fprintf(fp, "txl ");
      break;
   case nir_texop_txd:
      fprintf(fp, "txd ");
      break;
   case nir_texop_txf:
      fprintf(fp, "txf ");
      break;
   case nir_texop_txf_ms:
      fprintf(fp, "txf_ms ");
      break;
   case nir_texop_txf_ms_fb:
      fprintf(fp, "txf_ms_fb ");
      break;
   case nir_texop_txf_ms_mcs:
      fprintf(fp, "txf_ms_mcs ");
      break;
   case nir_texop_txs:
      fprintf(fp, "txs ");
      break;
   case nir_texop_lod:
      fprintf(fp, "lod ");
      break;
   case nir_texop_tg4:
      fprintf(fp, "tg4 ");
      break;
   case nir_texop_query_levels:
      fprintf(fp, "query_levels ");
      break;
   case nir_texop_texture_samples:
      fprintf(fp, "texture_samples ");
      break;
   case nir_texop_samples_identical:
      fprintf(fp, "samples_identical ");
      break;
   case nir_texop_tex_prefetch:
      fprintf(fp, "tex (pre-dispatchable) ");
      break;
   case nir_texop_fragment_fetch:
      fprintf(fp, "fragment_fetch ");
      break;
   case nir_texop_fragment_mask_fetch:
      fprintf(fp, "fragment_mask_fetch ");
      break;
   default:
      unreachable("Invalid texture operation");
      break;
   }

   bool has_texture_deref = false, has_sampler_deref = false;
   for (unsigned i = 0; i < instr->num_srcs; i++) {
      if (i > 0) {
         fprintf(fp, ", ");
      }

      print_src(&instr->src[i].src, state);
      fprintf(fp, " ");

      switch(instr->src[i].src_type) {
      case nir_tex_src_coord:
         fprintf(fp, "(coord)");
         break;
      case nir_tex_src_projector:
         fprintf(fp, "(projector)");
         break;
      case nir_tex_src_comparator:
         fprintf(fp, "(comparator)");
         break;
      case nir_tex_src_offset:
         fprintf(fp, "(offset)");
         break;
      case nir_tex_src_bias:
         fprintf(fp, "(bias)");
         break;
      case nir_tex_src_lod:
         fprintf(fp, "(lod)");
         break;
      case nir_tex_src_min_lod:
         fprintf(fp, "(min_lod)");
         break;
      case nir_tex_src_ms_index:
         fprintf(fp, "(ms_index)");
         break;
      case nir_tex_src_ms_mcs:
         fprintf(fp, "(ms_mcs)");
         break;
      case nir_tex_src_ddx:
         fprintf(fp, "(ddx)");
         break;
      case nir_tex_src_ddy:
         fprintf(fp, "(ddy)");
         break;
      case nir_tex_src_texture_deref:
         has_texture_deref = true;
         fprintf(fp, "(texture_deref)");
         break;
      case nir_tex_src_sampler_deref:
         has_sampler_deref = true;
         fprintf(fp, "(sampler_deref)");
         break;
      case nir_tex_src_texture_offset:
         fprintf(fp, "(texture_offset)");
         break;
      case nir_tex_src_sampler_offset:
         fprintf(fp, "(sampler_offset)");
         break;
      case nir_tex_src_texture_handle:
         fprintf(fp, "(texture_handle)");
         break;
      case nir_tex_src_sampler_handle:
         fprintf(fp, "(sampler_handle)");
         break;
      case nir_tex_src_plane:
         fprintf(fp, "(plane)");
         break;

      default:
         unreachable("Invalid texture source type");
         break;
      }
   }

   if (instr->op == nir_texop_tg4) {
      fprintf(fp, ", %u (gather_component)", instr->component);
   }

   if (nir_tex_instr_has_explicit_tg4_offsets(instr)) {
      fprintf(fp, ", { (%i, %i)", instr->tg4_offsets[0][0], instr->tg4_offsets[0][1]);
      for (unsigned i = 1; i < 4; ++i)
         fprintf(fp, ", (%i, %i)", instr->tg4_offsets[i][0],
                 instr->tg4_offsets[i][1]);
      fprintf(fp, " } (offsets)");
   }

   if (instr->op != nir_texop_txf_ms_fb) {
      if (!has_texture_deref) {
         fprintf(fp, ", %u (texture)", instr->texture_index);
      }

      if (!has_sampler_deref) {
         fprintf(fp, ", %u (sampler)", instr->sampler_index);
      }
   }

   if (instr->texture_non_uniform) {
      fprintf(fp, ", texture non-uniform");
   }

   if (instr->sampler_non_uniform) {
      fprintf(fp, ", sampler non-uniform");
   }

   if (instr->is_sparse) {
      fprintf(fp, ", sparse");
   }
}

static void
print_call_instr(nir_call_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, "call %s ", instr->callee->name);

   for (unsigned i = 0; i < instr->num_params; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      print_src(&instr->params[i], state);
   }
}

static void
print_load_const_instr(nir_load_const_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   print_ssa_def(&instr->def, state);

   fprintf(fp, " = load_const (");

   for (unsigned i = 0; i < instr->def.num_components; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      /*
       * we don't really know the type of the constant (if it will be used as a
       * float or an int), so just print the raw constant in hex for fidelity
       * and then print the float in a comment for readability.
       */

      switch (instr->def.bit_size) {
      case 64:
         fprintf(fp, "0x%16" PRIx64 " /* %f */", instr->value[i].u64,
                 instr->value[i].f64);
         break;
      case 32:
         fprintf(fp, "0x%08x /* %f */", instr->value[i].u32, instr->value[i].f32);
         break;
      case 16:
         fprintf(fp, "0x%04x /* %f */", instr->value[i].u16,
                 _mesa_half_to_float(instr->value[i].u16));
         break;
      case 8:
         fprintf(fp, "0x%02x", instr->value[i].u8);
         break;
      case 1:
         fprintf(fp, "%s", instr->value[i].b ? "true" : "false");
         break;
      }
   }

   fprintf(fp, ")");
}

static void
print_jump_instr(nir_jump_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   switch (instr->type) {
   case nir_jump_break:
      fprintf(fp, "break");
      break;

   case nir_jump_continue:
      fprintf(fp, "continue");
      break;

   case nir_jump_return:
      fprintf(fp, "return");
      break;

   case nir_jump_halt:
      fprintf(fp, "halt");
      break;

   case nir_jump_goto:
      fprintf(fp, "goto block_%u",
              instr->target ? instr->target->index : -1);
      break;

   case nir_jump_goto_if:
      fprintf(fp, "goto block_%u if ",
              instr->target ? instr->target->index : -1);
      print_src(&instr->condition, state);
      fprintf(fp, " else block_%u",
              instr->else_target ? instr->else_target->index : -1);
      break;
   }
}

static void
print_ssa_undef_instr(nir_ssa_undef_instr* instr, print_state *state)
{
   FILE *fp = state->fp;
   print_ssa_def(&instr->def, state);
   fprintf(fp, " = undefined");
}

static void
print_phi_instr(nir_phi_instr *instr, print_state *state)
{
   FILE *fp = state->fp;
   print_dest(&instr->dest, state);
   fprintf(fp, " = phi ");
   nir_foreach_phi_src(src, instr) {
      if (&src->node != exec_list_get_head(&instr->srcs))
         fprintf(fp, ", ");

      fprintf(fp, "block_%u: ", src->pred->index);
      print_src(&src->src, state);
   }
}

static void
print_parallel_copy_instr(nir_parallel_copy_instr *instr, print_state *state)
{
   FILE *fp = state->fp;
   nir_foreach_parallel_copy_entry(entry, instr) {
      if (&entry->node != exec_list_get_head(&instr->entries))
         fprintf(fp, "; ");

      print_dest(&entry->dest, state);
      fprintf(fp, " = ");
      print_src(&entry->src, state);
   }
}

static void
print_instr(const nir_instr *instr, print_state *state, unsigned tabs)
{
   FILE *fp = state->fp;
   print_tabs(tabs, fp);

   switch (instr->type) {
   case nir_instr_type_alu:
      print_alu_instr(nir_instr_as_alu(instr), state);
      break;

   case nir_instr_type_deref:
      print_deref_instr(nir_instr_as_deref(instr), state);
      break;

   case nir_instr_type_call:
      print_call_instr(nir_instr_as_call(instr), state);
      break;

   case nir_instr_type_intrinsic:
      print_intrinsic_instr(nir_instr_as_intrinsic(instr), state);
      break;

   case nir_instr_type_tex:
      print_tex_instr(nir_instr_as_tex(instr), state);
      break;

   case nir_instr_type_load_const:
      print_load_const_instr(nir_instr_as_load_const(instr), state);
      break;

   case nir_instr_type_jump:
      print_jump_instr(nir_instr_as_jump(instr), state);
      break;

   case nir_instr_type_ssa_undef:
      print_ssa_undef_instr(nir_instr_as_ssa_undef(instr), state);
      break;

   case nir_instr_type_phi:
      print_phi_instr(nir_instr_as_phi(instr), state);
      break;

   case nir_instr_type_parallel_copy:
      print_parallel_copy_instr(nir_instr_as_parallel_copy(instr), state);
      break;

   default:
      unreachable("Invalid instruction type");
      break;
   }
}

static int
compare_block_index(const void *p1, const void *p2)
{
   const nir_block *block1 = *((const nir_block **) p1);
   const nir_block *block2 = *((const nir_block **) p2);

   return (int) block1->index - (int) block2->index;
}

static void print_cf_node(nir_cf_node *node, print_state *state,
                          unsigned tabs);

static void
print_block(nir_block *block, print_state *state, unsigned tabs)
{
   FILE *fp = state->fp;

   print_tabs(tabs, fp);
   fprintf(fp, "block block_%u:\n", block->index);

   /* sort the predecessors by index so we consistently print the same thing */

   nir_block **preds =
      malloc(block->predecessors->entries * sizeof(nir_block *));

   unsigned i = 0;
   set_foreach(block->predecessors, entry) {
      preds[i++] = (nir_block *) entry->key;
   }

   qsort(preds, block->predecessors->entries, sizeof(nir_block *),
         compare_block_index);

   print_tabs(tabs, fp);
   fprintf(fp, "/* preds: ");
   for (unsigned i = 0; i < block->predecessors->entries; i++) {
      fprintf(fp, "block_%u ", preds[i]->index);
   }
   fprintf(fp, "*/\n");

   free(preds);

   nir_foreach_instr(instr, block) {
      print_instr(instr, state, tabs);
      fprintf(fp, "\n");
      print_annotation(state, instr);
   }

   print_tabs(tabs, fp);
   fprintf(fp, "/* succs: ");
   for (unsigned i = 0; i < 2; i++)
      if (block->successors[i]) {
         fprintf(fp, "block_%u ", block->successors[i]->index);
      }
   fprintf(fp, "*/\n");
}

static void
print_if(nir_if *if_stmt, print_state *state, unsigned tabs)
{
   FILE *fp = state->fp;

   print_tabs(tabs, fp);
   fprintf(fp, "if ");
   print_src(&if_stmt->condition, state);
   fprintf(fp, " {\n");
   foreach_list_typed(nir_cf_node, node, node, &if_stmt->then_list) {
      print_cf_node(node, state, tabs + 1);
   }
   print_tabs(tabs, fp);
   fprintf(fp, "} else {\n");
   foreach_list_typed(nir_cf_node, node, node, &if_stmt->else_list) {
      print_cf_node(node, state, tabs + 1);
   }
   print_tabs(tabs, fp);
   fprintf(fp, "}\n");
}

static void
print_loop(nir_loop *loop, print_state *state, unsigned tabs)
{
   FILE *fp = state->fp;

   print_tabs(tabs, fp);
   fprintf(fp, "loop {\n");
   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      print_cf_node(node, state, tabs + 1);
   }
   print_tabs(tabs, fp);
   fprintf(fp, "}\n");
}

static void
print_cf_node(nir_cf_node *node, print_state *state, unsigned int tabs)
{
   switch (node->type) {
   case nir_cf_node_block:
      print_block(nir_cf_node_as_block(node), state, tabs);
      break;

   case nir_cf_node_if:
      print_if(nir_cf_node_as_if(node), state, tabs);
      break;

   case nir_cf_node_loop:
      print_loop(nir_cf_node_as_loop(node), state, tabs);
      break;

   default:
      unreachable("Invalid CFG node type");
   }
}

static void
print_function_impl(nir_function_impl *impl, print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, "\nimpl %s ", impl->function->name);

   fprintf(fp, "{\n");

   nir_foreach_function_temp_variable(var, impl) {
      fprintf(fp, "\t");
      print_var_decl(var, state);
   }

   foreach_list_typed(nir_register, reg, node, &impl->registers) {
      fprintf(fp, "\t");
      print_register_decl(reg, state);
   }

   nir_index_blocks(impl);

   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      print_cf_node(node, state, 1);
   }

   fprintf(fp, "\tblock block_%u:\n}\n\n", impl->end_block->index);
}

static void
print_function(nir_function *function, print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, "decl_function %s (%d params)", function->name,
           function->num_params);

   fprintf(fp, "\n");

   if (function->impl != NULL) {
      print_function_impl(function->impl, state);
      return;
   }
}

static void
init_print_state(print_state *state, nir_shader *shader, FILE *fp)
{
   state->fp = fp;
   //state->fp = stderr;
   state->shader = shader;
   state->ht = _mesa_pointer_hash_table_create(NULL);
   state->syms = _mesa_set_create(NULL, _mesa_hash_string,
                                  _mesa_key_string_equal);
   state->index = 0;
}

static void
destroy_print_state(print_state *state)
{
   _mesa_hash_table_destroy(state->ht, NULL);
   _mesa_set_destroy(state->syms, NULL);
}

static const char *
primitive_name(unsigned primitive)
{
#define PRIM(X) case GL_ ## X : return #X
   switch (primitive) {
   PRIM(POINTS);
   PRIM(LINES);
   PRIM(LINE_LOOP);
   PRIM(LINE_STRIP);
   PRIM(TRIANGLES);
   PRIM(TRIANGLE_STRIP);
   PRIM(TRIANGLE_FAN);
   PRIM(QUADS);
   PRIM(QUAD_STRIP);
   PRIM(POLYGON);
   default:
      return "UNKNOWN";
   }
}


void
nir_print_shader_annotated(nir_shader *shader, FILE *fp,
                           struct hash_table *annotations)
{
   print_state state;
   init_print_state(&state, shader, fp);

   state.annotations = annotations;

   fprintf(fp, "shader: %s\n", gl_shader_stage_name(shader->info.stage));

   if (shader->info.name)
      fprintf(fp, "name: %s\n", shader->info.name);

   if (shader->info.label)
      fprintf(fp, "label: %s\n", shader->info.label);

   if (gl_shader_stage_is_compute(shader->info.stage)) {
      fprintf(fp, "local-size: %u, %u, %u%s\n",
              shader->info.cs.local_size[0],
              shader->info.cs.local_size[1],
              shader->info.cs.local_size[2],
              shader->info.cs.local_size_variable ? " (variable)" : "");
      fprintf(fp, "shared-size: %u\n", shader->info.cs.shared_size);
   }

   fprintf(fp, "inputs: %u\n", shader->num_inputs);
   fprintf(fp, "outputs: %u\n", shader->num_outputs);
   fprintf(fp, "uniforms: %u\n", shader->num_uniforms);
   if (shader->info.num_ubos)
      fprintf(fp, "ubos: %u\n", shader->info.num_ubos);
   fprintf(fp, "shared: %u\n", shader->shared_size);
   if (shader->scratch_size)
      fprintf(fp, "scratch: %u\n", shader->scratch_size);
   if (shader->constant_data_size)
      fprintf(fp, "constants: %u\n", shader->constant_data_size);

   if (shader->info.stage == MESA_SHADER_GEOMETRY) {
      fprintf(fp, "invocations: %u\n", shader->info.gs.invocations);
      fprintf(fp, "vertices in: %u\n", shader->info.gs.vertices_in);
      fprintf(fp, "vertices out: %u\n", shader->info.gs.vertices_out);
      fprintf(fp, "input primitive: %s\n", primitive_name(shader->info.gs.input_primitive));
      fprintf(fp, "output primitive: %s\n", primitive_name(shader->info.gs.output_primitive));
      fprintf(fp, "active_stream_mask: 0x%x\n", shader->info.gs.active_stream_mask);
      fprintf(fp, "uses_end_primitive: %u\n", shader->info.gs.uses_end_primitive);
   }

   nir_foreach_variable_in_shader(var, shader)
      print_var_decl(var, &state);

   foreach_list_typed(nir_function, func, node, &shader->functions) {
      print_function(func, &state);
   }

   destroy_print_state(&state);
}

void
nir_print_shader(nir_shader *shader, FILE *fp)
{
   nir_print_shader_annotated(shader, fp, NULL);
   fflush(fp);
}

void
nir_print_instr(const nir_instr *instr, FILE *fp)
{
   print_state state = {
      .fp = fp,
   };
   if (instr->block) {
      nir_function_impl *impl = nir_cf_node_get_function(&instr->block->cf_node);
      state.shader = impl->function->shader;
   }

   print_instr(instr, &state, 0);

}

void
nir_print_deref(const nir_deref_instr *deref, FILE *fp)
{
   print_state state = {
      .fp = fp,
   };
   print_deref_link(deref, true, &state);
}


// NIR to PTX translation below
typedef enum {
   UINT, // unsigned int
   INT,  // signed int
   FLOAT,
   BITS,
   PREDICATE,
   UNDEF
} val_type;


typedef struct  {
   int ssa_idx; // ssa_x
   int num_components; // vec 1,2,3,4
   int num_bits; // 1,8,32,64 bits
   val_type type; // int, float, or bool
   bool is_pointer;
   val_type pointer_type;
} ssa_reg_info;

static uint32_t loopID = 0;
static uint32_t ifID = 0;


static void
print_ptx_reg_decl(print_state *state, int vec_length, val_type type, int num_bits)
{
   FILE *fp = state->fp;
   fprintf(fp, ".reg ");

   if (vec_length == 2){
      fprintf(fp, ".v2 ");
   }
   else if (vec_length > 2 && vec_length <= 4){
      fprintf(fp, ".v4 ");
   }
   else if (vec_length > 4){
      abort();
   }

   switch (type) {
      case UINT:
         fprintf(fp, ".u%d", num_bits);
         break;
      case INT:
         fprintf(fp, ".s%d", num_bits);
         break;
      case FLOAT:
         fprintf(fp, ".f%d", num_bits);
         break;
      case BITS:
         fprintf(fp, ".b%d", num_bits); // i guess
         break;
      case PREDICATE:
         fprintf(fp, ".pred");
         break;
      case UNDEF:
         fprintf(fp, ".x%d", num_bits);
         break;
   }

   fprintf(fp, " ");
}


static void
print_ssa_def_as_ptx(nir_ssa_def *def, print_state *state, int position)
{
   FILE *fp = state->fp;
   if (def->name != NULL)
      fprintf(fp, "/* %s */ ", def->name);
   fprintf(fp, "%%ssa_%u", def->index);
   if (def->num_components > 1){
      switch (position) {
         case 0:
            fprintf(fp, ".x");
            break;
         case 1:
            fprintf(fp, ".y");
            break;
         case 2:
            fprintf(fp, ".z");
            break;
         case 3:
            fprintf(fp, ".w");
            break;
         case -1:
            break;
      }
   }
}


static void
print_ssa_use_as_ptx(nir_ssa_def *def, print_state *state)
{
   FILE *fp = state->fp;
   if (def->name != NULL)
      fprintf(fp, "/* %s */ ", def->name);
   fprintf(fp, "%%ssa_%u", def->index);
   /*if (def->num_components > 1){
      switch (position) {
         case 0:
            fprintf(fp, "X");
            break;
         case 1:
            fprintf(fp, "Y");
            break;
         case 2:
            fprintf(fp, "Z");
            break;
         case 3:
            fprintf(fp, "W");
            break;
         case -1:
            break;
      }
   }*/
}


static void
print_src_as_ptx(const nir_src *src, print_state *state)
{
   if (src->is_ssa)
      print_ssa_use_as_ptx(src->ssa, state);
   else
      print_reg_src(&src->reg, state);
}


static void
print_dest_as_ptx(nir_dest *dest, print_state *state, int position)
{
   if (dest->is_ssa)
      print_ssa_def_as_ptx(&dest->ssa, state, position);
   else
      print_reg_dest(&dest->reg, state);
}


static void
print_dest_as_ptx_no_pos(nir_dest *dest, print_state *state)
{
   if (dest->is_ssa)
      print_ssa_use_as_ptx(&dest->ssa, state);
   else
      print_reg_dest(&dest->reg, state);
}

static char*
val_type_to_str(val_type type)
{
   switch (type)
   {
   case UINT:
      return "u";
   case INT:
      return "s";
   case FLOAT:
      return "f";
   case BITS:
      return "b";
   case PREDICATE:
      return "pred";
   
   default:
      break;
   }
   return "";
}


static void
print_intrinsic_instr_as_ptx(nir_intrinsic_instr *instr, print_state *state, ssa_reg_info *ssa_register_info, unsigned tabs)
{
   const nir_intrinsic_info *info = &nir_intrinsic_infos[instr->intrinsic];
   unsigned num_srcs = info->num_srcs;
   FILE *fp = state->fp;

   // PTX Code

   //TODO: Double check all these data types

   if (!strcmp(info->name, "load_ray_launch_id")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = UINT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, UINT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_ray_launch_size")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = UINT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, UINT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "vulkan_resource_index")){
      // From nir_intrinsics.py
      /* # Vulkan descriptor set intrinsics
         #
         # The Vulkan API uses a different binding model from GL.  In the Vulkan
         # API, all external resources are represented by a tuple:
         #
         # (descriptor set, binding, array index)
         #
         # where the array index is the only thing allowed to be indirect.  The
         # vulkan_surface_index intrinsic takes the descriptor set and binding as
         # its first two indices and the array index as its source.  The third
         # index is a nir_variable_mode in case that's useful to the backend.
         #
         # The intended usage is that the shader will call vulkan_surface_index to
         # get an index and then pass that as the buffer index ubo/ssbo calls. */
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = UINT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, UINT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_vulkan_descriptor")){
      // I think it returns pointer to a member in the descriptor set
      // if the glsl layout is like this
      // layout(binding = 2, set = 0) uniform CameraProperties 
      // {
      //    mat4 viewInverse;
      //    mat4 projInverse;
      // } cam;
      // we can pass in the result from vulkan_resource_index to get the pointer to the cam struct
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = BITS;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, BITS, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_deref")){ // get address / pointer of a variable used for reading / loading
      if (info->has_dest) {
         val_type pointerType = FLOAT;
         if(ssa_register_info[instr->src[0].ssa->index].is_pointer)
            pointerType = ssa_register_info[instr->src[0].ssa->index].pointer_type;
         ssa_register_info[instr->dest.ssa.index].type = pointerType;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, pointerType, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s %d, ", info->name, instr->dest.ssa.num_components); // Intrinsic function name
   }
   
   // fprintf(fp, "%s%s, %d", is_parent_pointer ? ", ptr, " : ", not_ptr, ",
   //            glsl_get_struct_elem_name(parent->type, instr->strct.index), glsl_get_struct_field_offset(parent->type, instr->strct.index));

   else if (!strcmp(info->name, "store_deref")){ // get address / pointer of a variable used for writing / storing
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = FLOAT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
      // fprintf(fp, "%s%d, ", val_type_to_str(ssa_register_info[instr->src[1].ssa->index].type), ssa_register_info[instr->src[1].ssa->index].num_bits);
   }
   else if (!strcmp(info->name, "image_deref_load")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = FLOAT; // feel free to change the type, its used in the shader as imageLoad
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "image_deref_store")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = BITS;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, BITS, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "trace_ray")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = FLOAT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_ray_instance_custom_index")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = UINT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, UINT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_primitive_id")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = UINT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, UINT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_ray_world_to_object")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = FLOAT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_ray_object_to_world")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = FLOAT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_ray_world_direction")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = FLOAT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_ray_world_origin")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = FLOAT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_ray_t_max")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = FLOAT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_ray_t_min")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = FLOAT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "report_ray_intersection")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = PREDICATE;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, PREDICATE, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s.pred ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "shader_clock")){
      // The argument 2 probably means memory_scope=SUBGROUP
      // Store lower 32 bits in dst.x and upper 32 bits in dst.y
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = UINT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, UINT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_first_vertex")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = INT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, INT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_vertex_id_zero_base")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = INT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, INT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   // else if (!strcmp(info->name, "read_first_invocation")){
   //    int src_reg_idx = instr->src[0].ssa->index;
   //    val_type ssa_reg_type = ssa_register_info[src_reg_idx].type;
   //    if (info->has_dest) {
   //       // ssa_register_info[instr->dest.ssa.index].type = UINT;
   //       switch (ssa_reg_type)
   //       {
   //       case UINT:
   //          print_ptx_reg_decl(state, instr->dest.ssa.num_components, UINT, instr->dest.ssa.bit_size);
   //          break;
   //       case INT:
   //          print_ptx_reg_decl(state, instr->dest.ssa.num_components, INT, instr->dest.ssa.bit_size);
   //          break;
   //       case FLOAT:
   //          print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
   //          break;
   //       case BITS:
   //          print_ptx_reg_decl(state, instr->dest.ssa.num_components, BITS, instr->dest.ssa.bit_size);
   //          break;
   //       case PREDICATE:
   //          print_ptx_reg_decl(state, instr->dest.ssa.num_components, PREDICATE, instr->dest.ssa.bit_size);
   //          break;
   //       case UNDEF:
   //          printf("Should not be in here!\n");
   //          assert(0);
   //          break;
   //       }
   //       print_dest_as_ptx_no_pos(&instr->dest, state);
   //       fprintf(fp, ";\n");
   //       print_tabs(tabs, fp);
   //    }
   //    switch (ssa_reg_type) {
   //          case UINT:
   //             fprintf(fp, "mov.u%d ", instr->dest.ssa.bit_size);
   //             ssa_register_info[instr->dest.ssa.index].type = UINT;
   //             break;
   //          case INT:
   //             fprintf(fp, "mov.s%d ", instr->dest.ssa.bit_size);
   //             ssa_register_info[instr->dest.ssa.index].type = UINT;
   //             break;
   //          case FLOAT:
   //             fprintf(fp, "mov.f%d ", instr->dest.ssa.bit_size);
   //             ssa_register_info[instr->dest.ssa.index].type = UINT;
   //             break;
   //          case BITS:
   //             fprintf(fp, "mov.b%d ", instr->dest.ssa.bit_size);
   //             ssa_register_info[instr->dest.ssa.index].type = UINT;
   //             break;
   //          case PREDICATE:
   //             fprintf(fp, "mov.pred ");
   //             ssa_register_info[instr->dest.ssa.index].type = UINT;
   //             break;
   //          case UNDEF:
   //             printf("Should not be in here!\n");
   //             assert(0);
   //             break;
   //       }
   // }
   // else if (!strcmp(info->name, "reduce")){
   //    if (info->has_dest) {
   //       ssa_register_info[instr->dest.ssa.index].type = FLOAT;
   //       print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
   //       print_dest_as_ptx_no_pos(&instr->dest, state);
   //       fprintf(fp, ";\n");
   //       print_tabs(tabs, fp);
   //    }
   //    fprintf(fp, "%s ", info->name); // Intrinsic function name
   //    // fprintf(fp, "%s%d, ", val_type_to_str(ssa_register_info[instr->src[1].ssa->index].type), ssa_register_info[instr->src[1].ssa->index].num_bits);
   // }
   else if (!strcmp(info->name, "load_frag_coord")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = INT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_base_instance")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = INT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, INT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_instance_id")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = INT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, INT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "load_front_face")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = PREDICATE;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, PREDICATE, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else if (!strcmp(info->name, "discard_if")){
      if (info->has_dest) {
         ssa_register_info[instr->dest.ssa.index].type = INT;
         print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
         print_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
      }
      fprintf(fp, "%s ", info->name); // Intrinsic function name
   }
   else {
      fprintf(fp, "// untranslated %s instruction. ", info->name);
   }
         

   if (info->has_dest) {
      print_dest_as_ptx_no_pos(&instr->dest, state);
      if(num_srcs > 0)
         fprintf(fp, ", ");
   }

   for (unsigned i = 0; i < num_srcs; i++) {
      if (i != 0)
         fprintf(fp, ", ");
      // if (!strcmp(info->name, "read_first_invocation") && i == 1) {
      //    break;
      // }
      // if (!strcmp(info->name, "reduce") && i == 1) {
      //    break;
      // }

      print_src_as_ptx(&instr->src[i], state);
   }

   //fprintf(fp, ", ");

   for (unsigned i = 0; i < info->num_indices; i++) {
      if (!strcmp(info->name, "read_first_invocation")) {
         break;
      }
      if (!strcmp(info->name, "reduce") && i == 1) {
         break;
      }
      //if (i != 0)
      fprintf(fp, ", ");

      fprintf(fp, "%d", instr->const_index[i]);
   }

   fprintf(fp, ";");


   // NIR Code
   fprintf(fp, "\t// ");

   if (info->has_dest) {
      print_dest(&instr->dest, state);
      fprintf(fp, " = ");
   }

   fprintf(fp, "intrinsic %s (", info->name);

   for (unsigned i = 0; i < num_srcs; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      print_src_as_ptx(&instr->src[i], state);
   }

   fprintf(fp, ") (");

   for (unsigned i = 0; i < info->num_indices; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      fprintf(fp, "%d", instr->const_index[i]);
   }

   fprintf(fp, ")");

   for (unsigned i = 0; i < info->num_indices; i++) {
      unsigned idx = info->indices[i];
      fprintf(fp, " /*");
      switch (idx) {
      case NIR_INTRINSIC_WRITE_MASK: {
         /* special case wrmask to show it as a writemask.. */
         unsigned wrmask = nir_intrinsic_write_mask(instr);
         fprintf(fp, " wrmask=");
         for (unsigned i = 0; i < instr->num_components; i++)
            if ((wrmask >> i) & 1)
               fprintf(fp, "%c", comp_mask_string(instr->num_components)[i]);
         break;
      }

      case NIR_INTRINSIC_REDUCTION_OP: {
         nir_op reduction_op = nir_intrinsic_reduction_op(instr);
         fprintf(fp, " reduction_op=%s", nir_op_infos[reduction_op].name);
         break;
      }

      case NIR_INTRINSIC_IMAGE_DIM: {
         static const char *dim_name[] = {
            [GLSL_SAMPLER_DIM_1D] = "1D",
            [GLSL_SAMPLER_DIM_2D] = "2D",
            [GLSL_SAMPLER_DIM_3D] = "3D",
            [GLSL_SAMPLER_DIM_CUBE] = "Cube",
            [GLSL_SAMPLER_DIM_RECT] = "Rect",
            [GLSL_SAMPLER_DIM_BUF] = "Buf",
            [GLSL_SAMPLER_DIM_MS] = "2D-MSAA",
            [GLSL_SAMPLER_DIM_SUBPASS] = "Subpass",
            [GLSL_SAMPLER_DIM_SUBPASS_MS] = "Subpass-MSAA",
         };
         enum glsl_sampler_dim dim = nir_intrinsic_image_dim(instr);
         assert(dim < ARRAY_SIZE(dim_name) && dim_name[dim]);
         fprintf(fp, " image_dim=%s", dim_name[dim]);
         break;
      }

      case NIR_INTRINSIC_IMAGE_ARRAY: {
         bool array = nir_intrinsic_image_array(instr);
         fprintf(fp, " image_array=%s", array ? "true" : "false");
         break;
      }

      case NIR_INTRINSIC_FORMAT: {
         enum pipe_format format = nir_intrinsic_format(instr);
         fprintf(fp, " format=%s ", util_format_short_name(format));
         break;
      }

      case NIR_INTRINSIC_DESC_TYPE: {
         VkDescriptorType desc_type = nir_intrinsic_desc_type(instr);
         fprintf(fp, " desc_type=%s", vulkan_descriptor_type_name(desc_type));
         break;
      }

      case NIR_INTRINSIC_SRC_TYPE: {
         fprintf(fp, " src_type=");
         print_alu_type(nir_intrinsic_src_type(instr), state);
         break;
      }

      case NIR_INTRINSIC_DEST_TYPE: {
         fprintf(fp, " dest_type=");
         print_alu_type(nir_intrinsic_dest_type(instr), state);
         break;
      }

      case NIR_INTRINSIC_SWIZZLE_MASK: {
         fprintf(fp, " swizzle_mask=");
         unsigned mask = nir_intrinsic_swizzle_mask(instr);
         if (instr->intrinsic == nir_intrinsic_quad_swizzle_amd) {
            for (unsigned i = 0; i < 4; i++)
               fprintf(fp, "%d", (mask >> (i * 2) & 3));
         } else if (instr->intrinsic == nir_intrinsic_masked_swizzle_amd) {
            fprintf(fp, "((id & %d) | %d) ^ %d", mask & 0x1F,
                                                (mask >> 5) & 0x1F,
                                                (mask >> 10) & 0x1F);
         } else {
            fprintf(fp, "%d", mask);
         }
         break;
      }

      case NIR_INTRINSIC_MEMORY_SEMANTICS: {
         nir_memory_semantics semantics = nir_intrinsic_memory_semantics(instr);
         fprintf(fp, " mem_semantics=");
         switch (semantics & (NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE)) {
         case 0:                  fprintf(fp, "NONE");    break;
         case NIR_MEMORY_ACQUIRE: fprintf(fp, "ACQ");     break;
         case NIR_MEMORY_RELEASE: fprintf(fp, "REL");     break;
         default:                 fprintf(fp, "ACQ|REL"); break;
         }
         if (semantics & (NIR_MEMORY_MAKE_AVAILABLE)) fprintf(fp, "|AVAILABLE");
         if (semantics & (NIR_MEMORY_MAKE_VISIBLE))   fprintf(fp, "|VISIBLE");
         break;
      }

      case NIR_INTRINSIC_MEMORY_MODES: {
         fprintf(fp, " mem_modes=");
         unsigned int modes = nir_intrinsic_memory_modes(instr);
         while (modes) {
            nir_variable_mode m = u_bit_scan(&modes);
            fprintf(fp, "%s%s", get_variable_mode_str(1 << m, true), modes ? "|" : "");
         }
         break;
      }

      case NIR_INTRINSIC_EXECUTION_SCOPE:
      case NIR_INTRINSIC_MEMORY_SCOPE: {
         fprintf(fp, " %s=", nir_intrinsic_index_names[idx]);
         nir_scope scope =
            idx == NIR_INTRINSIC_MEMORY_SCOPE ? nir_intrinsic_memory_scope(instr)
                                              : nir_intrinsic_execution_scope(instr);
         switch (scope) {
         case NIR_SCOPE_NONE:         fprintf(fp, "NONE");         break;
         case NIR_SCOPE_DEVICE:       fprintf(fp, "DEVICE");       break;
         case NIR_SCOPE_QUEUE_FAMILY: fprintf(fp, "QUEUE_FAMILY"); break;
         case NIR_SCOPE_WORKGROUP:    fprintf(fp, "WORKGROUP");    break;
         case NIR_SCOPE_SHADER_CALL:  fprintf(fp, "SHADER_CALL");  break;
         case NIR_SCOPE_SUBGROUP:     fprintf(fp, "SUBGROUP");     break;
         case NIR_SCOPE_INVOCATION:   fprintf(fp, "INVOCATION");   break;
         }
         break;
      }

      case NIR_INTRINSIC_IO_SEMANTICS:
         fprintf(fp, " location=%u slots=%u",
                 nir_intrinsic_io_semantics(instr).location,
                 nir_intrinsic_io_semantics(instr).num_slots);
         if (state->shader) {
            if (state->shader->info.stage == MESA_SHADER_FRAGMENT &&
                instr->intrinsic == nir_intrinsic_store_output &&
                nir_intrinsic_io_semantics(instr).dual_source_blend_index) {
               fprintf(fp, " dualsrc=1");
            }
            if (state->shader->info.stage == MESA_SHADER_FRAGMENT &&
                instr->intrinsic == nir_intrinsic_load_output &&
                nir_intrinsic_io_semantics(instr).fb_fetch_output) {
               fprintf(fp, " fbfetch=1");
            }
            if (instr->intrinsic == nir_intrinsic_store_output &&
                nir_intrinsic_io_semantics(instr).per_view) {
               fprintf(fp, " perview=1");
            }
            if (state->shader->info.stage == MESA_SHADER_GEOMETRY &&
                instr->intrinsic == nir_intrinsic_store_output) {
               unsigned gs_streams = nir_intrinsic_io_semantics(instr).gs_streams;
               fprintf(fp, " gs_streams(");
               for (unsigned i = 0; i < 4; i++) {
                  fprintf(fp, "%s%c=%u", i ? " " : "", "xyzw"[i],
                          (gs_streams >> (i * 2)) & 0x3);
               }
               fprintf(fp, ")");
            }
            if (state->shader->info.stage == MESA_SHADER_FRAGMENT &&
                nir_intrinsic_io_semantics(instr).medium_precision) {
               fprintf(fp, " mediump");
            }
         }
         break;

      case NIR_INTRINSIC_ROUNDING_MODE: {
         fprintf(fp, " rounding_mode=");
         switch (nir_intrinsic_rounding_mode(instr)) {
         case nir_rounding_mode_undef: fprintf(fp, "undef");   break;
         case nir_rounding_mode_rtne:  fprintf(fp, "rtne");    break;
         case nir_rounding_mode_ru:    fprintf(fp, "ru");      break;
         case nir_rounding_mode_rd:    fprintf(fp, "rd");      break;
         case nir_rounding_mode_rtz:   fprintf(fp, "rtz");     break;
         default:                      fprintf(fp, "unkown");  break;
         }
         break;
      }

      default: {
         unsigned off = info->index_map[idx] - 1;
         fprintf(fp, " %s=%d", nir_intrinsic_index_names[idx], instr->const_index[off]);
         break;
      }
      }
      fprintf(fp, " */");
   }

   if (!state->shader)
      return;

   nir_variable_mode var_mode;
   switch (instr->intrinsic) {
   case nir_intrinsic_load_uniform:
      var_mode = nir_var_uniform;
      break;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_per_vertex_input:
      var_mode = nir_var_shader_in;
      break;
   case nir_intrinsic_load_output:
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
      var_mode = nir_var_shader_out;
      break;
   default:
      return;
   }

   nir_foreach_variable_with_modes(var, state->shader, var_mode) {
      if ((var->data.driver_location == nir_intrinsic_base(instr)) &&
          (instr->intrinsic == nir_intrinsic_load_uniform ||
           (nir_intrinsic_component(instr) >= var->data.location_frac  &&
            nir_intrinsic_component(instr) <
            (var->data.location_frac + glsl_get_components(var->type)))) &&
           var->name) {
         fprintf(fp, "\t/* %s */", var->name);
         break;
      }
   }
}


static void
print_type_decl(val_type ssa_reg_type, int num_bits, print_state *state)
{
   FILE *fp = state->fp;

   switch(ssa_reg_type) {
      case UINT:
         fprintf(fp, "u%d", num_bits);
         break;
      case INT:
         fprintf(fp, "s%d", num_bits);
         break;
      case FLOAT:
         fprintf(fp, "f%d", num_bits);
         break;
      case BITS:
         fprintf(fp, ".b%d", num_bits);
         break;
      case PREDICATE:
         fprintf(fp, ".pred");
         break;
      case UNDEF:
         printf("Should not be in here!\n");
         assert(0);
         break;
   }
}


static void
print_alu_src_as_ptx(nir_alu_instr *instr, unsigned src, print_state *state)
{
   FILE *fp = state->fp;

   if (instr->src[src].negate)
      fprintf(fp, "-");
   if (instr->src[src].abs)
      fprintf(fp, "abs(");

   print_src_as_ptx(&instr->src[src].src, state);

   bool print_swizzle = false;
   nir_component_mask_t used_channels = 0;

   for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++) { // this used to print out .xyzw
      if (!nir_alu_instr_channel_used(instr, src, i))
         continue;

      used_channels++;

      if (instr->src[src].swizzle[i] != i) {
         print_swizzle = true;
         break;
      }
   }

   unsigned live_channels = nir_src_num_components(instr->src[src].src);

   if (print_swizzle || used_channels != live_channels) {
      fprintf(fp, ".");
      for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++) {
         if (!nir_alu_instr_channel_used(instr, src, i))
            continue;

         fprintf(fp, "%c", comp_mask_string(live_channels)[instr->src[src].swizzle[i]]);
      }
   }

   if (instr->src[src].abs)
      fprintf(fp, ")");
}

static void
print_alu_dest_as_ptx(nir_alu_dest *dest, print_state *state, int position)
{
   FILE *fp = state->fp;
   /* we're going to print the saturate modifier later, after the opcode */

   print_dest_as_ptx(&dest->dest, state, position);

   if (!dest->dest.is_ssa &&
       dest->write_mask != (1 << dest->dest.reg.reg->num_components) - 1) {
      unsigned live_channels = dest->dest.reg.reg->num_components;
      fprintf(fp, ".");
      for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++)
         if ((dest->write_mask >> i) & 1)
            fprintf(fp, "%c", comp_mask_string(live_channels)[i]);
   }
}


static void
print_alu_dest_as_ptx_no_pos(nir_alu_dest *dest, print_state *state)
{
   FILE *fp = state->fp;
   /* we're going to print the saturate modifier later, after the opcode */

   print_dest_as_ptx_no_pos(&dest->dest, state);

   if (!dest->dest.is_ssa &&
       dest->write_mask != (1 << dest->dest.reg.reg->num_components) - 1) {
      unsigned live_channels = dest->dest.reg.reg->num_components;
      fprintf(fp, ".");
      for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++)
         if ((dest->write_mask >> i) & 1)
            fprintf(fp, "%c", comp_mask_string(live_channels)[i]);
   }
}


static void
print_alu_instr_as_ptx(nir_alu_instr *instr, print_state *state, ssa_reg_info *ssa_register_info, unsigned tabs)
{
   FILE *fp = state->fp;

   bool is_vec_type = (!strcmp(nir_op_infos[instr->op].name, "vec2") ||
                       !strcmp(nir_op_infos[instr->op].name, "vec3") || 
                       !strcmp(nir_op_infos[instr->op].name, "vec4"));

   // PTX here
   if (!is_vec_type) {
      // Opcodes
      if (!strcmp(nir_op_infos[instr->op].name, "u2f32")){
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "cvt.rn.f32");
         fprintf(fp, ".u%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "f2i32")){
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, INT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "cvt.rni.s32.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = INT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "i2f32")){
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "cvt.rn.f32");
         fprintf(fp, ".u%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "i2i64")){
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, INT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "cvt.s64.s32 ");

         ssa_register_info[instr->dest.dest.ssa.index].type = INT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "f2u32")){
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, UINT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "cvt.rni.u32.f32 ");

         ssa_register_info[instr->dest.dest.ssa.index].type = UINT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "b2f32")){
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "b2f32 ");

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fadd")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "add.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "frcp")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "rcp.approx.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fmul")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "mul.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "imul")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, INT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "mul.lo.s%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = INT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "inot")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, instr->dest.dest.ssa.bit_size == 1 ? PREDICATE : INT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         if (instr->dest.dest.ssa.bit_size == 1){
            fprintf(fp, "not.pred ");
         } else {
            fprintf(fp, "not.b%d ", instr->dest.dest.ssa.bit_size);
         }

         ssa_register_info[instr->dest.dest.ssa.index].type = instr->dest.dest.ssa.bit_size == 1 ? PREDICATE : INT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ior")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, instr->dest.dest.ssa.bit_size == 1 ? PREDICATE : UINT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         if (instr->dest.dest.ssa.bit_size == 1){
            fprintf(fp, "or.pred ");
         } else {
            fprintf(fp, "or.b%d ", instr->dest.dest.ssa.bit_size);
         }

         ssa_register_info[instr->dest.dest.ssa.index].type = instr->dest.dest.ssa.bit_size == 1 ? PREDICATE : UINT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ixor")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, instr->dest.dest.ssa.bit_size == 1 ? PREDICATE : UINT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         if (instr->dest.dest.ssa.bit_size == 1){
            fprintf(fp, "xor.pred ");
         } else {
            fprintf(fp, "xor.b%d ", instr->dest.dest.ssa.bit_size);
         }

         ssa_register_info[instr->dest.dest.ssa.index].type = instr->dest.dest.ssa.bit_size == 1 ? PREDICATE : UINT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "iand")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, instr->dest.dest.ssa.bit_size == 1 ? PREDICATE : UINT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         if (instr->dest.dest.ssa.bit_size == 1){
            fprintf(fp, "and.pred ");
         } else {
            fprintf(fp, "and.b%d ", instr->dest.dest.ssa.bit_size);
         }

         ssa_register_info[instr->dest.dest.ssa.index].type = instr->dest.dest.ssa.bit_size == 1 ? PREDICATE : UINT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ineg")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, INT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "neg.s%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = INT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "frsq")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "rsqrt.approx.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fneg")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "neg.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fmax")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "max.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fmin")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "min.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fabs")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "abs.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fpow")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "fpow ");

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "flrp")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "flrp ");

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ige")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "setp.ge.s%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ieq")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "setp.eq.s%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ine")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "setp.ne.s%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ilt")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "setp.lt.s%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ult")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "setp.lt.u%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "uge")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "setp.ge.u%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "flt")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "setp.lt.f%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fge")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "setp.ge.f%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "feq")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "setp.eq.f%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fsign")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "fsign ");

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fneu")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, PREDICATE, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "setp.ne.f%d ", instr->src[0].src.ssa->bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = PREDICATE;
      }

      else if (!strcmp(nir_op_infos[instr->op].name, "fsqrt")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "sqrt.approx.f%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "iadd")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, INT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "add.s%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = INT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ishl")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, INT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "shl.b%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = INT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ushr")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, UINT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "shr.u%d ", instr->dest.dest.ssa.bit_size);

         ssa_register_info[instr->dest.dest.ssa.index].type = UINT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fsat")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "fsat ");

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "mov")) { // need to get type of the src operands
         int src_reg_idx = instr->src[0].src.ssa->index;
         val_type ssa_reg_type = ssa_register_info[src_reg_idx].type;
         int num_bits = ssa_register_info[src_reg_idx].num_bits;

         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, ssa_reg_type, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         switch (ssa_reg_type) {
            case UINT:
               fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
               ssa_register_info[instr->dest.dest.ssa.index].type = UINT;
               break;
            case INT:
               fprintf(fp, "mov.s%d ", instr->dest.dest.ssa.bit_size);
               ssa_register_info[instr->dest.dest.ssa.index].type = UINT;
               break;
            case FLOAT:
               fprintf(fp, "mov.f%d ", instr->dest.dest.ssa.bit_size);
               ssa_register_info[instr->dest.dest.ssa.index].type = UINT;
               break;
            case BITS:
               fprintf(fp, "mov.b%d ", instr->dest.dest.ssa.bit_size);
               ssa_register_info[instr->dest.dest.ssa.index].type = UINT;
               break;
            case PREDICATE:
               fprintf(fp, "mov.pred ");
               ssa_register_info[instr->dest.dest.ssa.index].type = UINT;
               break;
            case UNDEF:
               printf("Should not be in here!\n");
               assert(0);
               break;
         }
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "bcsel")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);

         fprintf(fp, "bcsel ");

         ssa_register_info[instr->dest.dest.ssa.index].type = FLOAT;
      }
      // else if (!strcmp(nir_op_infos[instr->op].name, "pack_64_2x32_split")) {
      //    print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, UINT, instr->dest.dest.ssa.bit_size);
      //    print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      //    fprintf(fp, ";");
      //    fprintf(fp, "\n");
      //    print_tabs(tabs, fp);

      //    fprintf(fp, "pack_64_2x32_split ");

      //    ssa_register_info[instr->dest.dest.ssa.index].type = UINT;
      // }
      else if (!strcmp(nir_op_infos[instr->op].name, "ffloor")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT
         , instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "cvt.rzi.f32.f32 ");
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "ffract")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "cvt.rzi.f32.f32 ");
         // Prints the rest of the instruction
         for (unsigned i = 0; i < instr->dest.dest.ssa.num_components; i++) {
            // Src and Dst Operands
            print_alu_dest_as_ptx(&instr->dest, state, i);
            fprintf(fp, ", ");
            for (unsigned j = 0; j < nir_op_infos[instr->op].num_inputs; j++) {
               if (j != 0)
                  fprintf(fp, ", ");

               print_alu_src_as_ptx(instr, j, state);
            }
         }

         fprintf(fp, ";\n");
         print_tabs(tabs, fp);
         fprintf(fp, "sub.f32 ");
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fexp2")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "ex2.approx.f32 ");
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fsin")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "sin.approx.f32 ");
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fcos")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "cos.approx.f32 ");
      }
      // else if (!strcmp(nir_op_infos[instr->op].name, "ufind_msb")) {
      //    print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, INT, instr->dest.dest.ssa.bit_size);
      //    print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      //    fprintf(fp, ";");
      //    fprintf(fp, "\n");
      //    print_tabs(tabs, fp);
      //    fprintf(fp, "bfind.u32 ");
      //    // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      // }
      // else if (!strcmp(nir_op_infos[instr->op].name, "pack_half_2x16_split")) {
      //    print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
      //    print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      //    fprintf(fp, ";");
      //    fprintf(fp, "\n");
      //    print_tabs(tabs, fp);
      //    fprintf(fp, "pack_half_2x16_split ");
      //    // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      // }
      else if (!strcmp(nir_op_infos[instr->op].name, "extract_u16")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, UINT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "add.f32 ");
         // fprintf(fp, "extract_u16 ");
         // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fceil")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, UINT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "cvt.rpi.f32.f32 ");
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "flog2")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, UINT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "lg2.approx.f32 ");
      }
      // else if (!strcmp(nir_op_infos[instr->op].name, "imax")) {
      //    print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, INT, instr->dest.dest.ssa.bit_size);
      //    print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      //    fprintf(fp, ";");
      //    fprintf(fp, "\n");
      //    print_tabs(tabs, fp);
      //    fprintf(fp, "max.s32 ");
      //    // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      // }
      // else if (!strcmp(nir_op_infos[instr->op].name, "imin")) {
      //    print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, INT, instr->dest.dest.ssa.bit_size);
      //    print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      //    fprintf(fp, ";");
      //    fprintf(fp, "\n");
      //    print_tabs(tabs, fp);
      //    fprintf(fp, "min.s32 ");
      //    // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      // }
      // else if (!strcmp(nir_op_infos[instr->op].name, "bfi")) {
      //    print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, UINT, instr->dest.dest.ssa.bit_size);
      //    print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      //    fprintf(fp, ";");
      //    fprintf(fp, "\n");
      //    print_tabs(tabs, fp);
      //    fprintf(fp, "bfi_mesa ");
      //    // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      // }
      else if (!strcmp(nir_op_infos[instr->op].name, "fround_even")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "cvt.rni.f32.f32 ");
         // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "extract_u8")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, UINT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "extract_u8 ");
         // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fddy")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "fddy ");
         // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      }
      else if (!strcmp(nir_op_infos[instr->op].name, "fddx")) {
         print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
         print_alu_dest_as_ptx_no_pos(&instr->dest, state);
         fprintf(fp, ";");
         fprintf(fp, "\n");
         print_tabs(tabs, fp);
         fprintf(fp, "fddx ");
         // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      }
      // else if (!strcmp(nir_op_infos[instr->op].name, "bfm")) {
      //    print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, UINT, instr->dest.dest.ssa.bit_size);
      //    print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      //    fprintf(fp, ";");
      //    fprintf(fp, "\n");
      //    print_tabs(tabs, fp);
      //    fprintf(fp, "bfm ");
      //    // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      // }
      // else if (!strcmp(nir_op_infos[instr->op].name, "unpack_half_2x16_split_x")) {
      //    print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
      //    print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      //    fprintf(fp, ";");
      //    fprintf(fp, "\n");
      //    print_tabs(tabs, fp);
      //    fprintf(fp, "cvt.rz.f32.f16 ");
      //    // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      // }
      // else if (!strcmp(nir_op_infos[instr->op].name, "unpack_half_2x16_split_y")) {
      //    print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, FLOAT, instr->dest.dest.ssa.bit_size);
      //    print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      //    fprintf(fp, ";");
      //    fprintf(fp, "\n");
      //    print_tabs(tabs, fp);
      //    fprintf(fp, "shr.u32 ");
      //    // fprintf(fp, "mov.u%d ", instr->dest.dest.ssa.bit_size);
      // }
      else {
         fprintf(fp, "// Untranslated NIR instruction ");
      }

      // Prints the rest of the instruction
      for (unsigned i = 0; i < instr->dest.dest.ssa.num_components; i++) {
         // Src and Dst Operands
         print_alu_dest_as_ptx(&instr->dest, state, i);
         fprintf(fp, ", ");
         for (unsigned j = 0; j < nir_op_infos[instr->op].num_inputs; j++) {
            if (j != 0)
               fprintf(fp, ", ");

            print_alu_src_as_ptx(instr, j, state);
         }
      }
      if (!strcmp(nir_op_infos[instr->op].name, "ffract")) {
         fprintf(fp, ", ");
         print_alu_dest_as_ptx(&instr->dest, state, 0);
      }
      // if (!strcmp(nir_op_infos[instr->op].name, "unpack_half_2x16_split_y")) {
      //    fprintf(fp, ", 16 ");
      // }

      fprintf(fp, ";");
   }
   else { // Special case to handle vec2, vec3, etc...
     int src_reg_idx = instr->src[0].src.ssa->index;
     val_type ssa_reg_type = ssa_register_info[src_reg_idx].type;
     int num_bits = ssa_register_info[src_reg_idx].num_bits;

      print_ptx_reg_decl(state, instr->dest.dest.ssa.num_components, ssa_reg_type, instr->dest.dest.ssa.bit_size);
      print_alu_dest_as_ptx_no_pos(&instr->dest, state);
      fprintf(fp, ";\n");
      print_tabs(tabs, fp);
      for (unsigned i = 0; i < instr->dest.dest.ssa.num_components; i++) {
         if (i != 0) {
            fprintf(fp, "\n");
            print_tabs(tabs, fp);
         }

         //fprintf(fp, ".reg .f%d ", instr->dest.dest.ssa.bit_size);
         //print_alu_dest_as_ptx(&instr->dest, state, i);
         //fprintf(fp, "\n\t");

         fprintf(fp, "mov"); //TODO: set it as u32 for now, need to change later, prob record prev regs in a map and their type
      
         switch (ssa_reg_type) {
            case UINT:
               fprintf(fp, ".u%d ", num_bits);
               break;
            case INT:
               fprintf(fp, ".s%d ", num_bits);
               break;
            case FLOAT:
               fprintf(fp, ".f%d ", num_bits);
               break;
            case BITS:
               fprintf(fp, ".b%d ", num_bits); // i guess
               break;
            case PREDICATE:
               fprintf(fp, ".pred");
               break;
            case UNDEF:
               fprintf(fp, ".x%d ", num_bits);
               break;
         }

         print_alu_dest_as_ptx(&instr->dest, state, i);
         fprintf(fp, ", ");
         print_alu_src_as_ptx(instr, i, state);
         fprintf(fp, ";");
      }
   }


   // Original NIR
   fprintf(fp, "\t// ");
   print_alu_dest(&instr->dest, state);

   fprintf(fp, " = %s", nir_op_infos[instr->op].name);
   if (instr->exact)
      fprintf(fp, "!");
   if (instr->dest.saturate)
      fprintf(fp, ".sat");
   if (instr->no_signed_wrap)
      fprintf(fp, ".nsw");
   if (instr->no_unsigned_wrap)
      fprintf(fp, ".nuw");
   fprintf(fp, " ");

   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      print_alu_src(instr, i, state);
   }
}

static val_type 
glsl_base_type_to_val_type(enum glsl_base_type glsl_type)
{
   switch(glsl_type)
   {
      case GLSL_TYPE_UINT8:
      case GLSL_TYPE_UINT16:
      case GLSL_TYPE_UINT:
      case GLSL_TYPE_UINT64:
         return UINT;

      case GLSL_TYPE_INT8:
      case GLSL_TYPE_INT16:
      case GLSL_TYPE_INT:
      case GLSL_TYPE_INT64:
         return INT;

      case GLSL_TYPE_FLOAT:
      case GLSL_TYPE_FLOAT16:
      case GLSL_TYPE_DOUBLE:
         return FLOAT;

      case GLSL_TYPE_BOOL:
         return PREDICATE;
      
      case GLSL_TYPE_SAMPLER:
      case GLSL_TYPE_IMAGE:
         return BITS;
      
      case GLSL_TYPE_INTERFACE:
      case GLSL_TYPE_STRUCT:
      case GLSL_TYPE_ARRAY:
      case GLSL_TYPE_ATOMIC_UINT:
      case GLSL_TYPE_VOID:
      case GLSL_TYPE_SUBROUTINE:
      case GLSL_TYPE_FUNCTION:
      case GLSL_TYPE_ERROR:
      default:
         return UNDEF;
   }
}


static const char*
glsl_base_type_to_ptx_type(enum glsl_base_type glsl_type)
{
   switch(glsl_type)
   {
      case GLSL_TYPE_UINT:
         return "u32";
      case GLSL_TYPE_INT:
         return "s32";
      case GLSL_TYPE_FLOAT:
         return "f32";
      case GLSL_TYPE_FLOAT16:
         return "f16";
      case GLSL_TYPE_DOUBLE:
         return "f64";
      case GLSL_TYPE_UINT8:
         return "u8";
      case GLSL_TYPE_INT8:
         return "s8";
      case GLSL_TYPE_UINT16:
         return "u16";
      case GLSL_TYPE_INT16:
         return "s16";
      case GLSL_TYPE_UINT64:
         return "u64";
      case GLSL_TYPE_INT64:
         return "s64";
      case GLSL_TYPE_BOOL:
         return "b1";
      case GLSL_TYPE_SAMPLER:
      case GLSL_TYPE_IMAGE:
         return "descriptor";
      case GLSL_TYPE_INTERFACE:
      case GLSL_TYPE_STRUCT:
      case GLSL_TYPE_ARRAY:
      case GLSL_TYPE_ATOMIC_UINT:
      case GLSL_TYPE_VOID:
      case GLSL_TYPE_SUBROUTINE:
      case GLSL_TYPE_FUNCTION:
      case GLSL_TYPE_ERROR:
      default:
         return "xx";
   }
}

static void
print_deref_link_as_ptx(const nir_deref_instr *instr, bool whole_chain, print_state *state, ssa_reg_info *ssa_register_info)
{
   FILE *fp = state->fp;

   if (instr->deref_type == nir_deref_type_var) {
      fprintf(fp, "%s, %s", get_var_name(instr->var, state), glsl_base_type_to_ptx_type(glsl_get_base_type(instr->type)));
      // fprintf(fp, "%s", glsl_base_type_to_ptx_type(glsl_get_base_type(instr->type)));
      return;
   } else if (instr->deref_type == nir_deref_type_cast) {
      // fprintf(fp, "%s, ", glsl_get_type_name(instr->type));
      fprintf(fp, "%s, ", glsl_base_type_to_ptx_type(glsl_get_base_type(instr->type)));
      // For this type in NIR it prints something like CameraProperties*
      // However, we should put the size of the CameraProperties struct into a magic register and load the size from it
      // drop the * since its already a pointer type
      print_src_as_ptx(&instr->parent, state);
      return;
   }

   assert(instr->parent.is_ssa);
   nir_deref_instr *parent =
      nir_instr_as_deref(instr->parent.ssa->parent_instr);

   /* Is the parent we're going to print a bare cast? */
   const bool is_parent_cast =
      whole_chain && parent->deref_type == nir_deref_type_cast;

   /* If we're not printing the whole chain, the parent we print will be a SSA
    * value that represents a pointer.  The only deref type that naturally
    * gives a pointer is a cast.
    */
   const bool is_parent_pointer =
      !whole_chain || parent->deref_type == nir_deref_type_cast;

   /* Struct derefs have a nice syntax that works on pointers, arrays derefs
    * do not.
    */
   const bool need_deref =
      is_parent_pointer && instr->deref_type != nir_deref_type_struct;

   /* Cast need extra parens and so * dereferences */
   //if (is_parent_cast || need_deref)
   //   fprintf(fp, "(");

   if (need_deref){
      fprintf(fp, "1, "); // 1 means there was a star * for the variable
   }
   else {
      fprintf(fp, "0, ");
   }
      

   if (whole_chain) {
      print_deref_link_as_ptx(parent, whole_chain, state, ssa_register_info);
   } else {
      print_src_as_ptx(&instr->parent, state);
   }

   //if (is_parent_cast || need_deref)
   //   fprintf(fp, ")");

   fflush(fp);

   switch (instr->deref_type) {
   case nir_deref_type_struct:
      //fprintf(fp, "%s%s", is_parent_pointer ? "->" : ".",
      //        glsl_get_struct_elem_name(parent->type, instr->strct.index));
      fprintf(fp, "%s%s, %d, %s", is_parent_pointer ? ", ptr, " : ", not_ptr, ",
              glsl_get_struct_elem_name(parent->type, instr->strct.index), get_struct_field_offset_for_ptx(parent->type, instr->strct.index),
              glsl_base_type_to_ptx_type(glsl_get_base_type(instr->type)));
      break;

   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array: {
      if (nir_src_is_const(instr->arr.index)) {
         ssa_register_info->is_pointer = true;
         ssa_register_info->pointer_type = glsl_base_type_to_val_type(glsl_get_base_type(instr->type));
         //fprintf(fp, "[%"PRId64"]", nir_src_as_int(instr->arr.index));
         fprintf(fp, ", %"PRId64", %u, %s", nir_src_as_int(instr->arr.index), nir_deref_instr_array_stride(instr), 
            glsl_base_type_to_ptx_type(glsl_get_base_type(instr->type)));
      } else {
         ssa_register_info->is_pointer = true;
         ssa_register_info->pointer_type = glsl_base_type_to_val_type(glsl_get_base_type(instr->type));
         fprintf(fp, ", ");
         print_src_as_ptx(&instr->arr.index, state);
         
         fprintf(fp, ", %u, %s", nir_deref_instr_array_stride(instr), 
            glsl_base_type_to_ptx_type(glsl_get_base_type(instr->type)));
      }
      break;
   }

   case nir_deref_type_array_wildcard:
      fprintf(fp, "[*]");
      break;

   default:
      unreachable("Invalid deref instruction type");
   }
}


static void
print_deref_instr_as_ptx(nir_deref_instr *instr, print_state *state, ssa_reg_info *ssa_register_info)
{
   FILE *fp = state->fp;

   // PTX Code
   print_ptx_reg_decl(state, instr->dest.ssa.num_components, UINT, instr->dest.ssa.bit_size);
   print_dest_as_ptx_no_pos(&instr->dest, state);
   fprintf(fp, ";\n\t");

   ssa_register_info[instr->dest.ssa.index].type = BITS;
   ssa_register_info[instr->dest.ssa.index].num_bits = instr->dest.ssa.bit_size;
   ssa_register_info[instr->dest.ssa.index].num_components = instr->dest.ssa.num_components;
   ssa_register_info[instr->dest.ssa.index].ssa_idx = instr->dest.ssa.index;


   switch (instr->deref_type) {
   case nir_deref_type_var:
      fprintf(fp, "deref_var ");
      break;
   case nir_deref_type_array:
   case nir_deref_type_array_wildcard:
      fprintf(fp, "deref_array "); // get the pointer to the element in the array using the index in square brackets []
      break;
   case nir_deref_type_struct:
      fprintf(fp, "deref_struct "); // get the pointer to the member in the struct using the index in square brackets []
      break;
   case nir_deref_type_cast:
      fprintf(fp, "deref_cast "); // gets the type of the pointer
      break;
   case nir_deref_type_ptr_as_array:
      fprintf(fp, "deref_ptr_as_array ");
      break;
   default:
      unreachable("Invalid deref instruction type");
   }

   /* Only casts naturally return a pointer type */
   // if (instr->deref_type != nir_deref_type_cast)
   //    fprintf(fp, "&"); // TODO: this & gotta go

   print_dest_as_ptx_no_pos(&instr->dest, state);
   fprintf(fp, ", ");

   print_deref_link_as_ptx(instr, false, state, &ssa_register_info[instr->dest.ssa.index]);
   fprintf(fp, ", ");

   //fprintf(fp, " (");
   unsigned modes = instr->modes;
   // This prints out the type of thing its casting to, eg, ubo. We could also make ubo a spectial register where we do something about it
   while (modes) {
      int m = u_bit_scan(&modes);
      fprintf(fp, "%s%s", get_variable_mode_str(1 << m, true),
                          modes ? "|" : "");
   }
   //fprintf(fp, " %s) ", glsl_get_type_name(instr->type));
   fprintf(fp, ";");


   // Original NIR
   fprintf(fp, "\t// ");
   print_dest(&instr->dest, state);

   switch (instr->deref_type) {
   case nir_deref_type_var:
      fprintf(fp, " = deref_var ");
      break;
   case nir_deref_type_array:
   case nir_deref_type_array_wildcard:
      fprintf(fp, " = deref_array ");
      break;
   case nir_deref_type_struct:
      fprintf(fp, " = deref_struct ");
      break;
   case nir_deref_type_cast:
      fprintf(fp, " = deref_cast ");
      break;
   case nir_deref_type_ptr_as_array:
      fprintf(fp, " = deref_ptr_as_array ");
      break;
   default:
      unreachable("Invalid deref instruction type");
   }

   /* Only casts naturally return a pointer type */
   if (instr->deref_type != nir_deref_type_cast)
      fprintf(fp, "&");

   print_deref_link(instr, false, state);

   fprintf(fp, " (");
   /*unsigned*/ modes = instr->modes;
   while (modes) {
      int m = u_bit_scan(&modes);
      fprintf(fp, "%s%s", get_variable_mode_str(1 << m, true),
                          modes ? "|" : "");
   }
   fprintf(fp, " %s) ", glsl_get_type_name(instr->type));

   if (instr->deref_type != nir_deref_type_var &&
       instr->deref_type != nir_deref_type_cast) {
      /* Print the entire chain as a comment */
      fprintf(fp, "/* &");
      print_deref_link(instr, true, state);
      fprintf(fp, " */");
   }

   if (instr->deref_type == nir_deref_type_cast) {
      fprintf(fp, " /* ptr_stride=%u, align_mul=%u, align_offset=%u */",
              instr->cast.ptr_stride,
              instr->cast.align_mul, instr->cast.align_offset);
   }
}


static void
print_tex_instr_as_ptx(nir_tex_instr *instr, print_state *state, ssa_reg_info *ssa_register_info)
{
   FILE *fp = state->fp;

   // PTX Code
   print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
   // if (instr->op == nir_texop_tex) {
   //    print_ptx_reg_decl(state, 4, FLOAT, instr->dest.ssa.bit_size);
   // } else {
   //    print_ptx_reg_decl(state, instr->dest.ssa.num_components, FLOAT, instr->dest.ssa.bit_size);
   // }
   print_dest_as_ptx_no_pos(&instr->dest, state);
   fprintf(fp, ";\n\t");

   ssa_register_info[instr->dest.ssa.index].type = FLOAT;

   // assert(instr->op == nir_texop_txl);

   switch (instr->op) {
   case nir_texop_tex:
      fprintf(fp, "tex ");
      break;
   case nir_texop_txb:
      fprintf(fp, "txb ");
      break;
   case nir_texop_txl:
      fprintf(fp, "txl ");
      break;
   case nir_texop_txd:
      fprintf(fp, "txd ");
      break;
   case nir_texop_txf:
      fprintf(fp, "txf ");
      break;
   case nir_texop_txf_ms:
      fprintf(fp, "txf_ms ");
      break;
   case nir_texop_txf_ms_fb:
      fprintf(fp, "txf_ms_fb ");
      break;
   case nir_texop_txf_ms_mcs:
      fprintf(fp, "txf_ms_mcs ");
      break;
   case nir_texop_txs:
      fprintf(fp, "txs ");
      break;
   case nir_texop_lod:
      fprintf(fp, "lod ");
      break;
   case nir_texop_tg4:
      fprintf(fp, "tg4 ");
      break;
   case nir_texop_query_levels:
      fprintf(fp, "query_levels ");
      break;
   case nir_texop_texture_samples:
      fprintf(fp, "texture_samples ");
      break;
   case nir_texop_samples_identical:
      fprintf(fp, "samples_identical ");
      break;
   case nir_texop_tex_prefetch:
      fprintf(fp, "tex (pre-dispatchable) ");
      break;
   case nir_texop_fragment_fetch:
      fprintf(fp, "fragment_fetch ");
      break;
   case nir_texop_fragment_mask_fetch:
      fprintf(fp, "fragment_mask_fetch ");
      break;
   default:
      unreachable("Invalid texture operation");
      break;
   }

   print_dest_as_ptx_no_pos(&instr->dest, state);

   for (unsigned i = 0; i < instr->num_srcs; i++) {
      fprintf(fp, ", ");
      print_src_as_ptx(&instr->src[i].src, state);
   }

   fprintf(fp, ";");


   
   // Original NIR
   fprintf(fp, "\t// ");
   print_dest(&instr->dest, state);

   fprintf(fp, " = (");
   print_alu_type(instr->dest_type, state);
   fprintf(fp, ")");

   switch (instr->op) {
   case nir_texop_tex:
      fprintf(fp, "tex ");
      break;
   case nir_texop_txb:
      fprintf(fp, "txb ");
      break;
   case nir_texop_txl:
      fprintf(fp, "txl ");
      break;
   case nir_texop_txd:
      fprintf(fp, "txd ");
      break;
   case nir_texop_txf:
      fprintf(fp, "txf ");
      break;
   case nir_texop_txf_ms:
      fprintf(fp, "txf_ms ");
      break;
   case nir_texop_txf_ms_fb:
      fprintf(fp, "txf_ms_fb ");
      break;
   case nir_texop_txf_ms_mcs:
      fprintf(fp, "txf_ms_mcs ");
      break;
   case nir_texop_txs:
      fprintf(fp, "txs ");
      break;
   case nir_texop_lod:
      fprintf(fp, "lod ");
      break;
   case nir_texop_tg4:
      fprintf(fp, "tg4 ");
      break;
   case nir_texop_query_levels:
      fprintf(fp, "query_levels ");
      break;
   case nir_texop_texture_samples:
      fprintf(fp, "texture_samples ");
      break;
   case nir_texop_samples_identical:
      fprintf(fp, "samples_identical ");
      break;
   case nir_texop_tex_prefetch:
      fprintf(fp, "tex (pre-dispatchable) ");
      break;
   case nir_texop_fragment_fetch:
      fprintf(fp, "fragment_fetch ");
      break;
   case nir_texop_fragment_mask_fetch:
      fprintf(fp, "fragment_mask_fetch ");
      break;
   default:
      unreachable("Invalid texture operation");
      break;
   }

   bool has_texture_deref = false, has_sampler_deref = false;
   for (unsigned i = 0; i < instr->num_srcs; i++) {
      if (i > 0) {
         fprintf(fp, ", ");
      }

      print_src(&instr->src[i].src, state);
      fprintf(fp, " ");

      switch(instr->src[i].src_type) {
      case nir_tex_src_coord:
         fprintf(fp, "(coord)");
         break;
      case nir_tex_src_projector:
         fprintf(fp, "(projector)");
         break;
      case nir_tex_src_comparator:
         fprintf(fp, "(comparator)");
         break;
      case nir_tex_src_offset:
         fprintf(fp, "(offset)");
         break;
      case nir_tex_src_bias:
         fprintf(fp, "(bias)");
         break;
      case nir_tex_src_lod:
         fprintf(fp, "(lod)");
         break;
      case nir_tex_src_min_lod:
         fprintf(fp, "(min_lod)");
         break;
      case nir_tex_src_ms_index:
         fprintf(fp, "(ms_index)");
         break;
      case nir_tex_src_ms_mcs:
         fprintf(fp, "(ms_mcs)");
         break;
      case nir_tex_src_ddx:
         fprintf(fp, "(ddx)");
         break;
      case nir_tex_src_ddy:
         fprintf(fp, "(ddy)");
         break;
      case nir_tex_src_texture_deref:
         has_texture_deref = true;
         fprintf(fp, "(texture_deref)");
         break;
      case nir_tex_src_sampler_deref:
         has_sampler_deref = true;
         fprintf(fp, "(sampler_deref)");
         break;
      case nir_tex_src_texture_offset:
         fprintf(fp, "(texture_offset)");
         break;
      case nir_tex_src_sampler_offset:
         fprintf(fp, "(sampler_offset)");
         break;
      case nir_tex_src_texture_handle:
         fprintf(fp, "(texture_handle)");
         break;
      case nir_tex_src_sampler_handle:
         fprintf(fp, "(sampler_handle)");
         break;
      case nir_tex_src_plane:
         fprintf(fp, "(plane)");
         break;

      default:
         unreachable("Invalid texture source type");
         break;
      }
   }

   if (instr->op == nir_texop_tg4) {
      fprintf(fp, ", %u (gather_component)", instr->component);
   }

   if (nir_tex_instr_has_explicit_tg4_offsets(instr)) {
      fprintf(fp, ", { (%i, %i)", instr->tg4_offsets[0][0], instr->tg4_offsets[0][1]);
      for (unsigned i = 1; i < 4; ++i)
         fprintf(fp, ", (%i, %i)", instr->tg4_offsets[i][0],
                 instr->tg4_offsets[i][1]);
      fprintf(fp, " } (offsets)");
   }

   if (instr->op != nir_texop_txf_ms_fb) {
      if (!has_texture_deref) {
         fprintf(fp, ", %u (texture)", instr->texture_index);
      }

      if (!has_sampler_deref) {
         fprintf(fp, ", %u (sampler)", instr->sampler_index);
      }
   }

   if (instr->texture_non_uniform) {
      fprintf(fp, ", texture non-uniform");
   }

   if (instr->sampler_non_uniform) {
      fprintf(fp, ", sampler non-uniform");
   }

   if (instr->is_sparse) {
      fprintf(fp, ", sparse");
   }
}


static void
print_load_const_instr_as_ptx(nir_load_const_instr *instr, print_state *state, ssa_reg_info *ssa_register_info)
{
   FILE *fp = state->fp;

   // PTX here
   // Reg Decl

   int ptx_vec_len = 1;
   if (instr->def.num_components == 2){
      ptx_vec_len = 2;
   }
   else if (instr->def.num_components > 2 && instr->def.num_components <= 4){
      ptx_vec_len = 4;
   }
   else if (instr->def.num_components > 4){
      abort();
   }

   
   switch (instr->def.bit_size) {
      case 64:
         //fprintf(fp, ".reg .f64 "); // reg declaration
         print_ptx_reg_decl(state, instr->def.num_components, FLOAT, instr->def.bit_size);
         print_ssa_use_as_ptx(&instr->def, state);
         fprintf(fp, ";");
         ssa_register_info[instr->def.index].type = FLOAT;
         break;
      case 32:
         //fprintf(fp, ".reg .f32 "); // reg declaration
         print_ptx_reg_decl(state, instr->def.num_components, FLOAT, instr->def.bit_size);
         print_ssa_use_as_ptx(&instr->def, state);
         fprintf(fp, ";");
         ssa_register_info[instr->def.index].type = FLOAT;
         break;
      case 16:
         //fprintf(fp, ".reg .b16 "); // reg declaration
         print_ptx_reg_decl(state, instr->def.num_components, BITS, instr->def.bit_size);
         print_ssa_use_as_ptx(&instr->def, state);
         fprintf(fp, ";");
         ssa_register_info[instr->def.index].type = BITS;
         break;
      case 8:
         //fprintf(fp, ".reg .b8 "); // reg declaration
         print_ptx_reg_decl(state, instr->def.num_components, BITS, instr->def.bit_size);
         print_ssa_use_as_ptx(&instr->def, state);
         fprintf(fp, ";");
         ssa_register_info[instr->def.index].type = BITS;
         break;
      case 1:
         //fprintf(fp, ".reg .b1 "); // reg declaration
         print_ptx_reg_decl(state, instr->def.num_components, BITS, 32);
         print_ssa_use_as_ptx(&instr->def, state);
         fprintf(fp, ";");
         ssa_register_info[instr->def.index].type = BITS;
         break;
   }
   fprintf(fp, "\n\t");

   if(ptx_vec_len > 1)
   {
      for(int i = 0; i < ptx_vec_len; i++)
      {
         // print_ptx_reg_decl(state, 1, FLOAT, instr->def.bit_size);
         // print_ssa_use_as_ptx(&instr->def, state);
         // if(ptx_vec_len > 1)
         // {
         //    switch (i)
         //    {
         //    case 0:
         //       fprintf(fp, "x;");
         //       break;
            
         //    case 1:
         //       fprintf(fp, "y;");
         //       break;
            
         //    case 2:
         //       fprintf(fp, "z;");
         //       break;
            
         //    case 3:
         //       fprintf(fp, "w;");
         //       break;

         //    default:
         //       break;
         //    }
         // }


         // fprintf(fp, "\n\t");
         fprintf(fp, "mov");
         switch (instr->def.bit_size) {
         case 64:
            fprintf(fp, ".f64 ");
            print_ssa_use_as_ptx(&instr->def, state); //dst
            break;
         case 32:
            fprintf(fp, ".f32 ");
            print_ssa_use_as_ptx(&instr->def, state); //dst
            break;
         case 16:
            fprintf(fp, ".b16 ");
            print_ssa_use_as_ptx(&instr->def, state); //dst
            break;
         case 8:
            fprintf(fp, ".b8 ");
            print_ssa_use_as_ptx(&instr->def, state); //dst
            fprintf(fp, ", ");
            break;
         case 1:
            fprintf(fp, ".b1 ");
            print_ssa_use_as_ptx(&instr->def, state); //dst
            break;
         }

         fprintf(fp, "_%d", i);

         // switch (i)
         //    {
         //    case 0:
         //       fprintf(fp, "x");
         //       break;
            
         //    case 1:
         //       fprintf(fp, "y");
         //       break;
            
         //    case 2:
         //       fprintf(fp, "z");
         //       break;
            
         //    case 3:
         //       fprintf(fp, "w");
         //       break;

         //    default:
         //       break;
         //    }

         fprintf(fp, ", ");

         if (i > instr->def.num_components-1){
            switch (instr->def.bit_size) {
            case 64:
               fprintf(fp, "0D%#016" PRIx64, (uint64_t)0); // 0D stands for hex float representation
               break;
            case 32:
               fprintf(fp, "0F%08x", (uint32_t)0); // 0F stands for hex float representation
               break;
            case 16:
               fprintf(fp, "0x%04x /* %f */", (uint16_t)0,
                     _mesa_half_to_float((uint16_t)0));
               break;
            case 8:
               fprintf(fp, "0x%02x", (uint8_t)0);
               break;
            case 1:
               fprintf(fp, "%s", "0");
               break;
            }
         }
         else {
            switch (instr->def.bit_size) {
            case 64:
               fprintf(fp, "0D%#016" PRIx64, instr->value[i].u64); // 0D stands for hex float representation
               break;
            case 32:
               fprintf(fp, "0F%08x", instr->value[i].u32); // 0F stands for hex float representation
               break;
            case 16:
               fprintf(fp, "0x%04x /* %f */", instr->value[i].u16,
                     _mesa_half_to_float(instr->value[i].u16));
               break;
            case 8:
               fprintf(fp, "0x%02x", instr->value[i].u8);
               break;
            case 1:
               fprintf(fp, "%s", instr->value[i].b ? "1" : "0");
               break;
            }
         }

         fprintf(fp, ";");
         fprintf(fp, "\n\t");
      }
   }


   // Operand value in vectorized form
   if(ptx_vec_len == 1)
   {
      fprintf(fp, "load_const ");
      print_ssa_use_as_ptx(&instr->def, state);

      if (instr->def.num_components > 1) {
         fprintf(fp, ", {");
      }
      else {
         fprintf(fp, ", ");
      }

      for (unsigned i = 0; i < ptx_vec_len; i++) {
         if (i != 0) {
            fprintf(fp, ", ");
         }

         if(ptx_vec_len == 1) {
            if (i > instr->def.num_components-1){
               switch (instr->def.bit_size) {
               case 64:
                  fprintf(fp, "0D%#016" PRIx64, (uint64_t)0); // 0D stands for hex float representation
                  break;
               case 32:
                  fprintf(fp, "0F%08x", (uint32_t)0); // 0F stands for hex float representation
                  break;
               case 16:
                  fprintf(fp, "0x%04x /* %f */", (uint16_t)0,
                        _mesa_half_to_float((uint16_t)0));
                  break;
               case 8:
                  fprintf(fp, "0x%02x", (uint8_t)0);
                  break;
               case 1:
                  fprintf(fp, "%s", "0");
                  break;
               }
            }
            else {
               switch (instr->def.bit_size) {
               case 64:
                  fprintf(fp, "0D%#016" PRIx64, instr->value[i].u64); // 0D stands for hex float representation
                  break;
               case 32:
                  fprintf(fp, "0F%08x", instr->value[i].u32); // 0F stands for hex float representation
                  break;
               case 16:
                  fprintf(fp, "0x%04x /* %f */", instr->value[i].u16,
                        _mesa_half_to_float(instr->value[i].u16));
                  break;
               case 8:
                  fprintf(fp, "0x%02x", instr->value[i].u8);
                  break;
               case 1:
                  fprintf(fp, "%s", instr->value[i].b ? "1" : "0");
                  break;
               }
            }
         }
         else
         {
            print_ssa_use_as_ptx(&instr->def, state);
            switch (i)
            {
            case 0:
               fprintf(fp, "x");
               break;
            
            case 1:
               fprintf(fp, "y");
               break;
            
            case 2:
               fprintf(fp, "z");
               break;
            
            case 3:
               fprintf(fp, "w");
               break;

            default:
               break;
            }
         }
            
      }

      if (instr->def.num_components > 1) {
         fprintf(fp, "};");
      }
      else {
         fprintf(fp, ";");
      }
   }

   // for (unsigned i = 0; i < instr->def.num_components; i++) {
   //    if (i != 0) {
   //       fprintf(fp, "\t");
   //    }

   //    switch (instr->def.bit_size) {
   //    case 64:
   //       fprintf(fp, "\n\t");
   //       fprintf(fp, "mov.f64 ");
   //       print_ssa_def_as_ptx(&instr->def, state, i); //dst
   //       fprintf(fp, ", ");
   //       fprintf(fp, "0D%#016" PRIx64, instr->value[i].u64); // 0D stands for hex float representation
   //       break;
   //    case 32:
   //       fprintf(fp, "\n\t");
   //       fprintf(fp, "mov.f32 ");
   //       print_ssa_def_as_ptx(&instr->def, state, i); //dst
   //       fprintf(fp, ", ");
   //       fprintf(fp, "0F%08x", instr->value[i].u32); // 0F stands for hex float representation
   //       break;
   //    case 16:
   //       fprintf(fp, "\n\t");
   //       fprintf(fp, "mov.b16 ");
   //       print_ssa_def_as_ptx(&instr->def, state, i); //dst
   //       fprintf(fp, ", ");
   //       fprintf(fp, "0x%04x /* %f */", instr->value[i].u16,
   //               _mesa_half_to_float(instr->value[i].u16));
   //       break;
   //    case 8:
   //       fprintf(fp, "0x%02x", instr->value[i].u8);
   //       fprintf(fp, "\n\t");
   //       fprintf(fp, "mov.b8 ");
   //       print_ssa_def_as_ptx(&instr->def, state, i); //dst
   //       fprintf(fp, ", ");
   //       break;
   //    case 1:
   //       fprintf(fp, "\t");
   //       fprintf(fp, "mov.b1 ");
   //       print_ssa_def_as_ptx(&instr->def, state, i); //dst
   //       fprintf(fp, ", ");
   //       fprintf(fp, "%s", instr->value[i].b ? "1" : "0");
   //       break;
   //    }
   //    fprintf(fp, ";");
   // }
   

   // Original NIR
   fprintf(fp, "\t// "); //comment it out
   print_ssa_def(&instr->def, state);

   fprintf(fp, " = load_const (");

   for (unsigned i = 0; i < instr->def.num_components; i++) {
      if (i != 0)
         fprintf(fp, ", ");

      /*
       * we don't really know the type of the constant (if it will be used as a
       * float or an int), so just print the raw constant in hex for fidelity
       * and then print the float in a comment for readability.
       */

      switch (instr->def.bit_size) {
      case 64:
         fprintf(fp, "0x%16" PRIx64 " /* %f */", instr->value[i].u64,
                 instr->value[i].f64);
         break;
      case 32:
         fprintf(fp, "0x%08x /* %f */", instr->value[i].u32, instr->value[i].f32);
         break;
      case 16:
         fprintf(fp, "0x%04x /* %f */", instr->value[i].u16,
                 _mesa_half_to_float(instr->value[i].u16));
         break;
      case 8:
         fprintf(fp, "0x%02x", instr->value[i].u8);
         break;
      case 1:
         fprintf(fp, "%s", instr->value[i].b ? "true" : "false");
         break;
      }
   }

   fprintf(fp, ")");
}


static void
print_ssa_undef_instr_as_ptx(nir_ssa_undef_instr* instr, print_state *state, ssa_reg_info *ssa_register_info)
{
   FILE *fp = state->fp;

   // PTX Code
   print_ptx_reg_decl(state, instr->def.num_components, FLOAT, instr->def.bit_size);
   print_ssa_use_as_ptx(&instr->def, state);
   fprintf(fp, ";\n\t");

   fprintf(fp, "load_const ");
   print_ssa_use_as_ptx(&instr->def, state);
   fprintf(fp, ", 0F000000ff;");

   // Original NIR
   fprintf(fp, "\t// ");
   print_ssa_def(&instr->def, state);
   fprintf(fp, " = undefined");
}

static void
print_jump_instr_as_ptx(nir_jump_instr *instr, print_state *state)
{
   FILE *fp = state->fp;

   switch (instr->type) {
   case nir_jump_break:
      fprintf(fp, "bra loop_%d_exit;", loopID - 1);
      break;

   // case nir_jump_continue:
   //    fprintf(fp, "continue");
   //    break;

   case nir_jump_return:
      fprintf(fp, "return");
      break;

   case nir_jump_halt:
      fprintf(fp, "halt");
      break;

   case nir_jump_goto:
      fprintf(fp, "goto block_%u",
              instr->target ? instr->target->index : -1);
      break;

   case nir_jump_goto_if:
      fprintf(fp, "goto block_%u if ",
              instr->target ? instr->target->index : -1);
      print_src(&instr->condition, state);
      fprintf(fp, " else block_%u",
              instr->else_target ? instr->else_target->index : -1);
      break;
   }
}

static void
print_phi_instr_as_ptx(nir_phi_instr *instr, print_state *state, ssa_reg_info *ssa_register_info, unsigned tabs)
{
   FILE *fp = state->fp;

   val_type type = BITS; //TODO: fix this
   // nir_foreach_phi_src(src, instr) {
   //    if(type == UNDEF)
   //       type = ssa_register_info[src->src.ssa->index].type;
   //    else {
   //       if(type == ssa_register_info[src->src.ssa->index].type)
   //          continue;
   //       if(type == FLOAT) // todo: fix this. this happens because all load_consts are considered floats
   //          type = ssa_register_info[src->src.ssa->index].type;
   //    }
   // }

   print_ptx_reg_decl(state, instr->dest.ssa.num_components, type, instr->dest.ssa.bit_size);
   print_dest_as_ptx_no_pos(&instr->dest, state);
   fprintf(fp, ";");
   fprintf(fp, "\n");

   print_tabs(tabs, fp);
   fprintf(fp, "phi ");
   print_dest_as_ptx_no_pos(&instr->dest, state);
   fprintf(fp, ", ");
   nir_foreach_phi_src(src, instr) {
      if (&src->node != exec_list_get_head(&instr->srcs))
         fprintf(fp, ", ");

      fprintf(fp, "block_%u, ", src->pred->index);
      print_src_as_ptx(&src->src, state);
   }
   fprintf(fp, ";");

   ssa_register_info[instr->dest.ssa.index].type = type;
   ssa_register_info[instr->dest.ssa.index].num_components = instr->dest.ssa.num_components;
   ssa_register_info[instr->dest.ssa.index].num_bits = instr->dest.ssa.bit_size;
   ssa_register_info[instr->dest.ssa.index].ssa_idx = instr->dest.ssa.index;

   // Original NIR
   print_tabs(tabs, fp);
   fprintf(fp, "// ");

   print_dest(&instr->dest, state);
   fprintf(fp, " = phi ");
   nir_foreach_phi_src(src, instr) {
      if (&src->node != exec_list_get_head(&instr->srcs))
         fprintf(fp, ", ");

      fprintf(fp, "block_%u: ", src->pred->index);
      print_src(&src->src, state);
   }
}

static void
print_instr_as_ptx(const nir_instr *instr, print_state *state, unsigned tabs, ssa_reg_info *ssa_register_info)
{
   FILE *fp = state->fp;
   print_tabs(tabs, fp);

   switch (instr->type) {
   case nir_instr_type_alu: ;
      nir_alu_instr *alu_instr = nir_instr_as_alu(instr);
      if (alu_instr->dest.dest.is_ssa) {
         ssa_register_info[alu_instr->dest.dest.ssa.index].ssa_idx = alu_instr->dest.dest.ssa.index;
         ssa_register_info[alu_instr->dest.dest.ssa.index].num_components = (int) alu_instr->dest.dest.ssa.num_components;
         ssa_register_info[alu_instr->dest.dest.ssa.index].num_bits = (int) alu_instr->dest.dest.ssa.bit_size;
      }
      print_alu_instr_as_ptx(nir_instr_as_alu(instr), state, ssa_register_info, tabs);
      break;

   case nir_instr_type_deref:
      print_deref_instr_as_ptx(nir_instr_as_deref(instr), state, ssa_register_info);
      break;

   case nir_instr_type_call:
      assert(0);
      print_call_instr(nir_instr_as_call(instr), state);
      break;

   case nir_instr_type_intrinsic: ;
      nir_intrinsic_instr *intrinsic_instr = nir_instr_as_intrinsic(instr);
      if (intrinsic_instr->dest.is_ssa) {
         ssa_register_info[intrinsic_instr->dest.ssa.index].ssa_idx = intrinsic_instr->dest.ssa.index;
         ssa_register_info[intrinsic_instr->dest.ssa.index].num_components = (int) intrinsic_instr->dest.ssa.num_components;
         ssa_register_info[intrinsic_instr->dest.ssa.index].num_bits = (int) intrinsic_instr->dest.ssa.bit_size;
      }
      print_intrinsic_instr_as_ptx(nir_instr_as_intrinsic(instr), state, ssa_register_info, tabs);
      break;

   case nir_instr_type_tex:
      print_tex_instr_as_ptx(nir_instr_as_tex(instr), state, ssa_register_info);
      break;

   case nir_instr_type_load_const: ;
      nir_load_const_instr *load_const_instr = nir_instr_as_load_const(instr);
      ssa_register_info[load_const_instr->def.index].ssa_idx = load_const_instr->def.index;
      ssa_register_info[load_const_instr->def.index].num_components = (int) load_const_instr->def.num_components;
      ssa_register_info[load_const_instr->def.index].num_bits = (int) load_const_instr->def.bit_size;
      print_load_const_instr_as_ptx(nir_instr_as_load_const(instr), state, ssa_register_info);
      break;

   case nir_instr_type_jump:
      print_jump_instr_as_ptx(nir_instr_as_jump(instr), state);
      break;

   case nir_instr_type_ssa_undef: ;
      nir_ssa_undef_instr * ssa_undef_instr = nir_instr_as_ssa_undef(instr);
      ssa_register_info[ssa_undef_instr->def.index].ssa_idx = ssa_undef_instr->def.index;
      ssa_register_info[ssa_undef_instr->def.index].num_components = (int) ssa_undef_instr->def.num_components;
      ssa_register_info[ssa_undef_instr->def.index].num_bits = (int) ssa_undef_instr->def.bit_size;
      ssa_register_info[ssa_undef_instr->def.index].type = FLOAT;
      print_ssa_undef_instr_as_ptx(nir_instr_as_ssa_undef(instr), state, ssa_register_info);
      break;

   case nir_instr_type_phi:
      print_phi_instr_as_ptx(nir_instr_as_phi(instr), state, ssa_register_info, tabs);
      break;

   case nir_instr_type_parallel_copy:
      assert(0);
      print_parallel_copy_instr(nir_instr_as_parallel_copy(instr), state);
      break;

   default:
      unreachable("Invalid instruction type");
      break;
   }
   fprintf(fp, "\n"); // For easier readability
}


static void
print_block_as_ptx(nir_block *block, print_state *state, ssa_reg_info *ssa_register_info, unsigned tabs)
{
   FILE *fp = state->fp;

   print_tabs(tabs, fp);
   fprintf(fp, "// start_block block_%u:\n", block->index);

   /* sort the predecessors by index so we consistently print the same thing */

   nir_block **preds =
      malloc(block->predecessors->entries * sizeof(nir_block *));

   unsigned i = 0;
   set_foreach(block->predecessors, entry) {
      preds[i++] = (nir_block *) entry->key;
   }

   qsort(preds, block->predecessors->entries, sizeof(nir_block *),
         compare_block_index);

   print_tabs(tabs, fp);
   fprintf(fp, "// preds: ");
   for (unsigned i = 0; i < block->predecessors->entries; i++) {
      fprintf(fp, "block_%u ", preds[i]->index);
   }
   fprintf(fp, "\n");

   free(preds);

   // Pre-parse the ssa register number to make the register info table
   int instr_count = 0;
   nir_foreach_instr(instr, block) {
      instr_count++;
   }

   // Printing PTX
   nir_foreach_instr(instr, block) {
      print_instr_as_ptx(instr, state, tabs, ssa_register_info);
      fprintf(fp, "\n");
      print_annotation(state, instr);
   }

   print_tabs(tabs, fp);
   fprintf(fp, "// succs: ");
   for (unsigned i = 0; i < 2; i++)
      if (block->successors[i]) {
         fprintf(fp, "block_%u ", block->successors[i]->index);
      }
   fprintf(fp, "\n");

   print_tabs(tabs, fp);
   fprintf(fp, "// end_block block_%u:\n", block->index);
}

static void print_cf_node_as_ptx(nir_cf_node *node, print_state *state, ssa_reg_info *ssa_register_info, unsigned int tabs);

static void
print_loop_as_ptx(nir_loop *loop, print_state *state, ssa_reg_info *ssa_register_info, unsigned tabs)
{
   FILE *fp = state->fp;

   uint32_t currentID = loopID++;
   print_tabs(tabs, fp);
   fprintf(fp, "loop_%d: \n", currentID);
   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      print_cf_node_as_ptx(node, state, ssa_register_info, tabs + 1);
   }
   print_tabs(tabs + 1, fp);
   fprintf(fp, "bra loop_%d;\n", currentID);
   print_tabs(tabs, fp);
   fprintf(fp, "\n");
   print_tabs(tabs, fp);
   fprintf(fp, "loop_%d_exit:\n", currentID);
}

static void
print_if_as_ptx(nir_if *if_stmt, print_state *state, ssa_reg_info *ssa_register_info, unsigned tabs)
{
   FILE *fp = state->fp;

   uint32_t currentID = ifID++;
   print_tabs(tabs, fp);
   fprintf(fp, "//if\n");

   print_tabs(tabs, fp);
   fprintf(fp, "@!");
   print_src_as_ptx(&if_stmt->condition, state);
   fprintf(fp, " bra else_%d;\n", currentID);

   print_tabs(tabs, fp);
   fprintf(fp, "\n");
   
   foreach_list_typed(nir_cf_node, node, node, &if_stmt->then_list) {
      print_cf_node_as_ptx(node, state, ssa_register_info, tabs + 1);
   }
   print_tabs(tabs + 1, fp);
   fprintf(fp, "bra end_if_%d;\n", currentID);
   
   print_tabs(tabs, fp);
   fprintf(fp, "\n");
   print_tabs(tabs, fp);
   fprintf(fp, "else_%d: \n", currentID);

   foreach_list_typed(nir_cf_node, node, node, &if_stmt->else_list) {
      print_cf_node_as_ptx(node, state, ssa_register_info, tabs + 1);
   }

   print_tabs(tabs, fp);
   fprintf(fp, "end_if_%d:\n", currentID);
}

static void
print_cf_node_as_ptx(nir_cf_node *node, print_state *state, ssa_reg_info *ssa_register_info, unsigned int tabs)
{
   switch (node->type) {
   case nir_cf_node_block:
      print_block_as_ptx(nir_cf_node_as_block(node), state, ssa_register_info, tabs);
      break;

   case nir_cf_node_if:
      print_if_as_ptx(nir_cf_node_as_if(node), state, ssa_register_info, tabs);
      break;

   case nir_cf_node_loop:
      print_loop_as_ptx(nir_cf_node_as_loop(node), state, ssa_register_info, tabs);
      break;

   default:
      unreachable("Invalid CFG node type");
   }
}

// static val_type
// glsl_base_type_to_val_type(enum glsl_base_type type)
// {
//    switch (type)
//    {
//    case GLSL_TYPE_UINT:
//    case GLSL_TYPE_UINT8:
//    case GLSL_TYPE_UINT16:
//    case GLSL_TYPE_UINT64:
//       return UINT;
   
//    case GLSL_TYPE_INT:
//    case GLSL_TYPE_INT8:
//    case GLSL_TYPE_INT16:
//    case GLSL_TYPE_INT64:
//       return INT;
   
//    case GLSL_TYPE_FLOAT:
//    case GLSL_TYPE_FLOAT16:
//       return FLOAT;
   
//    case GLSL_TYPE_BOOL:
//    case GLSL_TYPE_IMAGE:
//       return BITS;


//    case GLSL_TYPE_SAMPLER:
//    case GLSL_TYPE_ATOMIC_UINT:
//    case GLSL_TYPE_STRUCT:
//    case GLSL_TYPE_INTERFACE:
//    case GLSL_TYPE_ARRAY:
//    case GLSL_TYPE_VOID:
//    case GLSL_TYPE_SUBROUTINE:
//    case GLSL_TYPE_FUNCTION:
//    case GLSL_TYPE_ERROR:
//    case GLSL_TYPE_DOUBLE:
//    default:
//       assert(0);
//       break;
//    }
// }

// static uint32_t
// glsl_base_type_to_num_bits(enum glsl_base_type type)
// {
//    switch (type)
//    {
//    case GLSL_TYPE_BOOL:
//       return 1;
   
//    case GLSL_TYPE_UINT8:
//    case GLSL_TYPE_INT8:
//       return 8;
   
//    case GLSL_TYPE_UINT16:
//    case GLSL_TYPE_INT16:
//    case GLSL_TYPE_FLOAT16:
//       return 16;
   
//    case GLSL_TYPE_UINT:
//    case GLSL_TYPE_INT:
//    case GLSL_TYPE_FLOAT:
//       return 32;
   
//    case GLSL_TYPE_UINT64:
//    case GLSL_TYPE_INT64:
//       return 64;

//    case GLSL_TYPE_SAMPLER:
//    case GLSL_TYPE_IMAGE:
//    case GLSL_TYPE_ATOMIC_UINT:
//    case GLSL_TYPE_STRUCT:
//    case GLSL_TYPE_INTERFACE:
//    case GLSL_TYPE_ARRAY:
//    case GLSL_TYPE_VOID:
//    case GLSL_TYPE_SUBROUTINE:
//    case GLSL_TYPE_FUNCTION:
//    case GLSL_TYPE_ERROR:
//    case GLSL_TYPE_DOUBLE:
//       assert(0);
//       return 0;
   
//    default:
//       break;
//    }
// }


// static void
// print_ptx_local_decl(print_state *state, val_type type, int num_bits, char* name, int length)
// {
//    FILE *fp = state->fp;
//    fprintf(fp, ".local ");

//    switch (type) {
//       case UINT:
//          fprintf(fp, ".u%d", num_bits);
//          break;
//       case INT:
//          fprintf(fp, ".s%d", num_bits);
//          break;
//       case FLOAT:
//          fprintf(fp, ".f%d", num_bits);
//          break;
//       case BITS:
//          fprintf(fp, ".b%d", num_bits); // i guess
//          break;
//       case UNDEF: // ignore this
//          break;
//    }

//    fprintf(fp, " %s", name);

//    if (length > 1) {
//       fprintf(fp, "[%d]", length);
//    }
//    fprintf(fp, ";\t");
// }

static void
print_var_decl_as_ptx(nir_variable *var, print_state *state)
{
   FILE *fp = state->fp;

   // PTX Code
   // The variables here are probably treated as magic variables
   unsigned size = 0;
   unsigned align = 0;

   glsl_get_natural_size_align_bytes(var->type, &size, &align);

   //TODO: check to see if the new one above creates different resutls
   // This is what used to work forthings other than arrays
   // if(glsl_get_base_type(var->type) == GLSL_TYPE_STRUCT)
   //    size = get_struct_size_for_ptx(var->type);
   // else
   //    size = glsl_get_bit_size(var->type) / 8;
   
   // if(glsl_get_bit_size(var->type) % 8 != 0)
   //    size++;
   

   // else if (glsl_get_base_type(var->type) == GLSL_TYPE_ARRAY)
   // {
   //    glsl_array_size(var->type);
   //    printf("this is where things go wrong\n");
   // }
   
   if(size < 4)
      size = 4;

   const char *loc = NULL;
   char buf[4];
   const char *components = NULL;
   if (var->data.mode == nir_var_shader_in ||
       var->data.mode == nir_var_shader_out ||
       var->data.mode == nir_var_uniform ||
       var->data.mode == nir_var_mem_ubo ||
       var->data.mode == nir_var_mem_ssbo)
   {

      switch (state->shader->info.stage)
      {
      case MESA_SHADER_VERTEX:
         if (var->data.mode == nir_var_shader_in)
            loc = gl_vert_attrib_name(var->data.location);
         else if (var->data.mode == nir_var_shader_out)
            loc = gl_varying_slot_name_for_stage(var->data.location,
                                                 state->shader->info.stage);
         break;
      case MESA_SHADER_GEOMETRY:
         if ((var->data.mode == nir_var_shader_in) ||
             (var->data.mode == nir_var_shader_out))
         {
            loc = gl_varying_slot_name_for_stage(var->data.location,
                                                 state->shader->info.stage);
         }
         break;
      case MESA_SHADER_FRAGMENT:
         if (var->data.mode == nir_var_shader_in)
         {
            loc = gl_varying_slot_name_for_stage(var->data.location,
                                                 state->shader->info.stage);
         }
         else if (var->data.mode == nir_var_shader_out)
         {
            loc = gl_frag_result_name(var->data.location);
         }
         break;
      case MESA_SHADER_TESS_CTRL:
      case MESA_SHADER_TESS_EVAL:
      case MESA_SHADER_COMPUTE:
      case MESA_SHADER_KERNEL:
      default:
         /* TODO */
         break;
      }

      if (!loc)
      {
         if (var->data.location == ~0)
         {
            loc = "~0";
         }
         else
         {
            snprintf(buf, sizeof(buf), "%u", var->data.location);
            loc = buf;
         }
      }

      /* For shader I/O vars that have been split to components or packed,
       * print the fractional location within the input/output.
       */
      unsigned int num_components =
          glsl_get_components(glsl_without_array(var->type));
      char components_local[18] = {'_' /* the rest is 0-filled */};
      switch (var->data.mode)
      {
      case nir_var_shader_in:
      case nir_var_shader_out:
         if (num_components < 16 && num_components != 0)
         {
            const char *xyzw = comp_mask_string(num_components);
            for (int i = 0; i < num_components; i++)
               components_local[i + 1] = xyzw[i + var->data.location_frac];

            components = components_local;
         }
         break;
      default:
         break;
      }
   }
   fprintf(fp, "decl_var %s, %d, %d, %d, %d, %u, %u, %s%s;\t", get_var_name(var, state), size, glsl_get_vector_elements(var->type), 
                  glsl_get_base_type(var->type), var->data.mode, var->data.descriptor_set, var->data.binding, loc? loc : "UNDEFINED", components ? components : "");

   // if ((var->data.mode == nir_var_shader_temp) ||
   //       (var->data.mode == nir_var_shader_call_data) ||
   //       (var->data.mode == nir_var_ray_hit_attrib) ||
   //       (var->data.mode == nir_var_mem_constant))
   // {
   //    print_ptx_local_decl(state, glsl_base_type_to_val_type(glsl_get_base_type(var->type)),
   //                         glsl_get_bit_size(var->type), var->name, glsl_get_vector_elements(var->type));
   // }

   // Original NIR
   fprintf(fp, "// decl_var ");

   const char *const cent = (var->data.centroid) ? "centroid " : "";
   const char *const samp = (var->data.sample) ? "sample " : "";
   const char *const patch = (var->data.patch) ? "patch " : "";
   const char *const inv = (var->data.invariant) ? "invariant " : "";
   const char *const per_view = (var->data.per_view) ? "per_view " : "";
   fprintf(fp, "%s%s%s%s%s%s %s ",
           cent, samp, patch, inv, per_view,
           get_variable_mode_str(var->data.mode, false),
           glsl_interp_mode_name(var->data.interpolation));

   enum gl_access_qualifier access = var->data.access;
   const char *const coher = (access & ACCESS_COHERENT) ? "coherent " : "";
   const char *const volat = (access & ACCESS_VOLATILE) ? "volatile " : "";
   const char *const restr = (access & ACCESS_RESTRICT) ? "restrict " : "";
   const char *const ronly = (access & ACCESS_NON_WRITEABLE) ? "readonly " : "";
   const char *const wonly = (access & ACCESS_NON_READABLE) ? "writeonly " : "";
   const char *const reorder = (access & ACCESS_CAN_REORDER) ? "reorderable " : "";
   fprintf(fp, "%s%s%s%s%s%s", coher, volat, restr, ronly, wonly, reorder);

   if (glsl_get_base_type(glsl_without_array(var->type)) == GLSL_TYPE_IMAGE) {
      fprintf(fp, "%s ", util_format_short_name(var->data.image.format));
   }

   if (var->data.precision) {
      const char *precisions[] = {
         "",
         "highp",
         "mediump",
         "lowp",
      };
      fprintf(fp, "%s ", precisions[var->data.precision]);
   }

   fprintf(fp, "%s %s", glsl_get_type_name(var->type),
           get_var_name(var, state));


      fprintf(fp, " (%s%s, %u, %u)%s", loc,
              components ? components : "",
              var->data.descriptor_set, var->data.binding,
            //   var->data.driver_location, var->data.binding,
              var->data.compact ? " compact" : "");

   if (var->constant_initializer) {
      fprintf(fp, " = { ");
      print_constant(var->constant_initializer, var->type, state);
      fprintf(fp, " }");
   }
   if (glsl_type_is_sampler(var->type) && var->data.sampler.is_inline_sampler) {
      fprintf(fp, " = { %s, %s, %s }",
              get_constant_sampler_addressing_mode(var->data.sampler.addressing_mode),
              var->data.sampler.normalized_coordinates ? "true" : "false",
              get_constant_sampler_filter_mode(var->data.sampler.filter_mode));
   }
   if (var->pointer_initializer)
      fprintf(fp, " = &%s", get_var_name(var->pointer_initializer, state));

   fprintf(fp, "\n");
   print_annotation(state, var);
}

static uint32_t functionID = 0; //TODO: when shaders are registered, function name is calculated seperately, combine them into one ID

static void
print_ptx_function_impl(nir_function_impl *impl, print_state *state, gl_shader_stage stage)
{
   FILE *fp = state->fp;
   uint32_t id = 0;

   //fprintf(fp, "\n// impl %s \n", impl->function->name);

   fprintf(fp, ".entry ");

   switch (stage) {
      case MESA_SHADER_RAYGEN:
         fprintf(fp, "MESA_SHADER_RAYGEN");
         id = functionID;
         break;
      case MESA_SHADER_ANY_HIT:
         fprintf(fp, "MESA_SHADER_ANY_HIT");
         id = functionID;
         break;
      case MESA_SHADER_CLOSEST_HIT:
         fprintf(fp, "MESA_SHADER_CLOSEST_HIT");
         id = functionID;
         break;
      case MESA_SHADER_MISS:
         fprintf(fp, "MESA_SHADER_MISS");
         id = functionID;
         break;
      case MESA_SHADER_INTERSECTION:
         fprintf(fp, "MESA_SHADER_INTERSECTION");
         id = functionID;
         break;
      case MESA_SHADER_CALLABLE:
         fprintf(fp, "MESA_SHADER_CALLABLE");
         id = functionID;
         break;
      case MESA_SHADER_VERTEX:
         fprintf(fp, "MESA_SHADER_VERTEX");
         id = functionID - 1;
         break;
      case MESA_SHADER_FRAGMENT:
         fprintf(fp, "MESA_SHADER_FRAGMENT");
         id = functionID + 1;
         break;
      default:
         unreachable("Invalid shader type");
   }

   functionID++;
   fprintf(fp, "_func%d", id++);

   fprintf(fp, "_%s ", impl->function->name);
   fprintf(fp, "() "); // any shader inputs would go here

   fprintf(fp, "{\n");

   nir_foreach_function_temp_variable(var, impl) {
      fprintf(fp, "\t");
      print_var_decl_as_ptx(var, state);
   }

   foreach_list_typed(nir_register, reg, node, &impl->registers) {
      fprintf(fp, "\t");
      print_register_decl(reg, state);
   }

   nir_index_blocks(impl);

   ssa_reg_info *ssa_register_info = malloc(sizeof(*ssa_register_info) * 5000);
   memset(ssa_register_info, 0, sizeof(*ssa_register_info) * 5000);

   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      print_cf_node_as_ptx(node, state, ssa_register_info, 1);
   }

   free(ssa_register_info);

   fprintf(fp, "\t// block block_%u:\n", impl->end_block->index);

   fprintf(fp, "\tshader_exit:\n");
   fprintf(fp, "\texit;\n");
   fprintf(fp, "}\n");
}


static void
print_ptx_function(nir_function *function, print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, "// decl_function %s (%d params)", function->name,
           function->num_params);

   fprintf(fp, "\n");

   if (function->impl != NULL) {
      print_ptx_function_impl(function->impl, state, function->shader->info.stage);
      return;
   }
}


static void
print_ptx_header(print_state *state)
{
   FILE *fp = state->fp;

   fprintf(fp, ".version 2.0\n");
   fprintf(fp, ".target sm_10, map_f64_to_f32\n");

   fprintf(fp, "\n");
}


void
nir_translate_shader_annotated(nir_shader *shader, FILE *fp,
                               struct hash_table *annotations)
{
   print_state state;
   init_print_state(&state, shader, fp);

   state.annotations = annotations;

   print_ptx_header(&state);

   fprintf(fp, "// shader: %s\n", gl_shader_stage_name(shader->info.stage));

   if (shader->info.name)
      fprintf(fp, "// name: %s\n", shader->info.name);

   if (shader->info.label)
      fprintf(fp, "// label: %s\n", shader->info.label);

   if (gl_shader_stage_is_compute(shader->info.stage)) {
      fprintf(fp, "local-size: %u, %u, %u%s\n",
              shader->info.cs.local_size[0],
              shader->info.cs.local_size[1],
              shader->info.cs.local_size[2],
              shader->info.cs.local_size_variable ? " (variable)" : "");
      fprintf(fp, "shared-size: %u\n", shader->info.cs.shared_size);
   }

   fprintf(fp, "// inputs: %u\n", shader->num_inputs);
   fprintf(fp, "// outputs: %u\n", shader->num_outputs);
   fprintf(fp, "// uniforms: %u\n", shader->num_uniforms);
   if (shader->info.num_ubos)
      fprintf(fp, "// ubos: %u\n", shader->info.num_ubos);
   fprintf(fp, "// shared: %u\n", shader->shared_size);
   if (shader->scratch_size)
      fprintf(fp, "// scratch: %u\n", shader->scratch_size);
   if (shader->constant_data_size)
      fprintf(fp, "// constants: %u\n", shader->constant_data_size);

   if (shader->info.stage == MESA_SHADER_GEOMETRY) {
      fprintf(fp, "invocations: %u\n", shader->info.gs.invocations);
      fprintf(fp, "vertices in: %u\n", shader->info.gs.vertices_in);
      fprintf(fp, "vertices out: %u\n", shader->info.gs.vertices_out);
      fprintf(fp, "input primitive: %s\n", primitive_name(shader->info.gs.input_primitive));
      fprintf(fp, "output primitive: %s\n", primitive_name(shader->info.gs.output_primitive));
      fprintf(fp, "active_stream_mask: 0x%x\n", shader->info.gs.active_stream_mask);
      fprintf(fp, "uses_end_primitive: %u\n", shader->info.gs.uses_end_primitive);
   }

   nir_foreach_variable_in_shader(var, shader)
      print_var_decl_as_ptx(var, &state);

   foreach_list_typed(nir_function, func, node, &shader->functions) {
      print_ptx_function(func, &state);
   }

   destroy_print_state(&state);
}

void
nir_translate_shader_to_ptx(nir_shader *shader, FILE *fp, char *filePath)
{
   nir_translate_shader_annotated(shader, fp, NULL);
   fflush(fp);
}