// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "error.h"
#include "texture.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// ============================================================================
// CPU Host Texture Implementation
// ============================================================================

using wp::get_texture_bytes_per_channel;
using wp::Texture;

uint64_t wp_texture_create_host(
    int ndim,
    int num_mip_levels,
    int* mip_widths,
    int* mip_heights,
    int* mip_depths,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    void** mip_data_ptrs_out
)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    if (num_mip_levels < 1 || num_mip_levels > WP_TEXTURE_MAX_MIP_LEVELS) {
        wp::set_error_string(
            "Warp error: Number of texture mip levels must be in [1, %d], got %d", WP_TEXTURE_MAX_MIP_LEVELS,
            num_mip_levels
        );
        return 0;
    }

    for (int i = 0; i < num_mip_levels; i++) {
        if (mip_widths[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture width must be a positive integer at mip level %d, got %d", i, mip_widths[i]
            );
            return 0;
        }
        if (ndim > 1 && mip_heights[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture height must be a positive integer at mip level %d, got %d", i, mip_heights[i]
            );
            return 0;
        }
        if (ndim > 2 && mip_depths[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture depth must be a positive integer at mip level %d, got %d", i, mip_depths[i]
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    int32_t addr_u = address_modes[0];
    int32_t addr_v = ndim > 1 ? address_modes[1] : 0;
    int32_t addr_w = ndim > 2 ? address_modes[2] : 0;

    Texture* tex = new (wp_alloc_host(sizeof(Texture), "(native:texture)")) Texture(
        ndim, num_mip_levels, mip_widths, mip_heights, mip_depths, num_channels, dtype, filter_mode, mip_filter_mode,
        addr_u, addr_v, addr_w, use_normalized_coords
    );

    if (mip_data_ptrs_out) {
        for (int i = 0; i < num_mip_levels; i++) {
            mip_data_ptrs_out[i] = tex->mip_data[i];
        }
    }

    return reinterpret_cast<uint64_t>(tex);
}

void wp_texture_destroy_host(uint64_t tex_handle)
{
    Texture* tex = (Texture*)tex_handle;
    tex->~Texture();
    wp_free_host(tex);
}

#if WP_ENABLE_CUDA || WP_ENABLE_HIP

// ============================================================================
// CUDA Device Texture Implementation
// ============================================================================

#include "cuda_util.h"

#if WP_ENABLE_HIP
#define WP_TEXTURE_HIP 1
constexpr int WP_TEXTURE_MEMORY_HOST = 1;
constexpr int WP_TEXTURE_MEMORY_DEVICE = 2;
constexpr int WP_TEXTURE_MEMORY_ARRAY = 3;
#else
#define WP_TEXTURE_HIP 0
#endif

// Helper function to get CUDA array format from dtype
static CUarray_format get_cuda_format(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return CU_AD_FORMAT_UNSIGNED_INT8;
    case WP_TEXTURE_DTYPE_UINT16:
        return CU_AD_FORMAT_UNSIGNED_INT16;
    case WP_TEXTURE_DTYPE_UINT32:
        return CU_AD_FORMAT_UNSIGNED_INT32;
    case WP_TEXTURE_DTYPE_INT8:
        return CU_AD_FORMAT_SIGNED_INT8;
    case WP_TEXTURE_DTYPE_INT16:
        return CU_AD_FORMAT_SIGNED_INT16;
    case WP_TEXTURE_DTYPE_INT32:
        return CU_AD_FORMAT_SIGNED_INT32;
    case WP_TEXTURE_DTYPE_FLOAT16:
        return CU_AD_FORMAT_HALF;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return CU_AD_FORMAT_FLOAT;
    }
}

// Helper function to convert address mode int to CUDA address mode
#if WP_TEXTURE_HIP
static hipTextureAddressMode get_cuda_address_mode(int address_mode)
#else
static CUaddress_mode get_cuda_address_mode(int address_mode)
#endif
{
    switch (address_mode) {
    case 0:
#if WP_TEXTURE_HIP
        return hipAddressModeWrap;
#else
        return CU_TR_ADDRESS_MODE_WRAP;
#endif
    case 1:
#if WP_TEXTURE_HIP
        return hipAddressModeClamp;
#else
        return CU_TR_ADDRESS_MODE_CLAMP;
#endif
    case 2:
#if WP_TEXTURE_HIP
        return hipAddressModeMirror;
#else
        return CU_TR_ADDRESS_MODE_MIRROR;
#endif
    case 3:
#if WP_TEXTURE_HIP
        return hipAddressModeBorder;
#else
        return CU_TR_ADDRESS_MODE_BORDER;
#endif
    default:
#if WP_TEXTURE_HIP
        return hipAddressModeClamp;
#else
        return CU_TR_ADDRESS_MODE_CLAMP;
#endif
    }
}

uint64_t wp_texture_create_device(
    void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access, int num_mip_levels
)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture dimensions must be positive integers, got %d at dimension %d", shape[i], i
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    if (num_mip_levels < 1 || num_mip_levels > WP_TEXTURE_MAX_MIP_LEVELS) {
        wp::set_error_string(
            "Warp error: Number of texture mip levels must be in [1, %d], got %d", WP_TEXTURE_MAX_MIP_LEVELS,
            num_mip_levels
        );
        return 0;
    }

    ContextGuard guard(context);

    size_t width = shape[0];
    size_t height = ndim > 1 ? shape[1] : 0;
    size_t depth = ndim > 2 ? shape[2] : 0;

#if WP_TEXTURE_HIP
    hipDevice_t device;
    hipDeviceProp_t device_properties;
    if (!check_cuda(hipGetDevice(&device)) || !check_cuda(hipGetDeviceProperties(&device_properties, device))) {
        return 0;
    }

    const size_t dimensions[] = {width, height, depth};
    for (int i = 0; i < ndim; i++) {
        if (dimensions[i] > static_cast<size_t>(device_properties.maxTexture3D[i])) {
            wp::set_error_string(
                "Warp error: HIP texture dimension %d (%zu) exceeds device limit (%d)", i, dimensions[i],
                device_properties.maxTexture3D[i]
            );
            return 0;
        }
    }
#endif

#if WP_TEXTURE_HIP
    hipArray_Format format = get_cuda_format(dtype);
    HIP_ARRAY3D_DESCRIPTOR arr_desc = {};
#else
    CUarray_format format = get_cuda_format(dtype);
    CUDA_ARRAY3D_DESCRIPTOR arr_desc = {};
#endif
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Depth = depth;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;
    arr_desc.Flags = surface_access ? CUDA_ARRAY3D_SURFACE_LDST : 0;

    if (num_mip_levels > 1) {
#if WP_TEXTURE_HIP
        hipMipmappedArray_t mipmap_array = nullptr;
        if (!check_cuda(hipMipmappedArrayCreate(&mipmap_array, &arr_desc, static_cast<unsigned int>(num_mip_levels)))) {
#else
        CUmipmappedArray mipmap_array = NULL;
        if (!check_cu(cuMipmappedArrayCreate_f(&mipmap_array, &arr_desc, (unsigned int)num_mip_levels))) {
#endif
            return 0;
        }
        return reinterpret_cast<uint64_t>(mipmap_array);
    }

#if WP_TEXTURE_HIP
    hipArray_t cuda_array = nullptr;
    if (!check_cuda(hipArray3DCreate(&cuda_array, &arr_desc))) {
#else
    CUarray cuda_array = NULL;
    if (!check_cu(cuArray3DCreate_f(&cuda_array, &arr_desc))) {
#endif
        return 0;
    }
    return reinterpret_cast<uint64_t>(cuda_array);
}

void wp_texture_destroy_device(void* context, uint64_t array_handle, bool is_mipmapped)
{
    release_or_defer(is_mipmapped ? ReleaseKind::MipmappedArray : ReleaseKind::Array, context, array_handle);
}

uint64_t wp_texture_get_mip_level_array_device(void* context, uint64_t mipmap_array_handle, int level)
{
    if (!mipmap_array_handle) {
        wp::set_error_string("Warp error: Null mipmapped array handle");
        return 0;
    }

    ContextGuard guard(context);

#if WP_TEXTURE_HIP
    hipArray_t level_array = nullptr;
    if (!check_cuda(hipMipmappedArrayGetLevel(&level_array, reinterpret_cast<hipMipmappedArray_t>(static_cast<uintptr_t>(mipmap_array_handle)), static_cast<unsigned int>(level)))) {
#else
    CUarray level_array = NULL;
    if (!check_cu(
            cuMipmappedArrayGetLevel_f(&level_array, (CUmipmappedArray)mipmap_array_handle, (unsigned int)level)
        )) {
#endif
        return 0;
    }
    return reinterpret_cast<uint64_t>(level_array);
}

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned width,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);

#if WP_TEXTURE_HIP
    hipMemcpy3DParms copy_params = {};
    // HIP requires non-zero height/depth for 1D and 2D copies as well.
    const bool copy_from_host = src_memory_type == WP_TEXTURE_MEMORY_HOST;
    const bool copy_to_host = dst_memory_type == WP_TEXTURE_MEMORY_HOST;
    const bool dst_is_array = dst_memory_type == WP_TEXTURE_MEMORY_ARRAY;
    const bool src_is_array = src_memory_type == WP_TEXTURE_MEMORY_ARRAY;
    copy_params.extent = make_hipExtent(
        dst_is_array || src_is_array ? width : width_bytes, height ? height : 1, depth ? depth : 1
    );
    copy_params.kind = copy_from_host ? hipMemcpyHostToDevice
        : (copy_to_host ? hipMemcpyDeviceToHost : hipMemcpyDeviceToDevice);
#else
    CUDA_MEMCPY3D copy_params = {};
    copy_params.WidthInBytes = width_bytes;
    copy_params.Height = height;
    copy_params.Depth = depth;
#endif

#if WP_TEXTURE_HIP
    if (dst_memory_type == WP_TEXTURE_MEMORY_HOST) {
        copy_params.dstPtr = make_hipPitchedPtr(
            reinterpret_cast<void*>(static_cast<uintptr_t>(dst_handle)), dst_pitch, width_bytes, dst_height ? dst_height : 1
        );
    } else if (dst_memory_type == WP_TEXTURE_MEMORY_DEVICE) {
        copy_params.dstPtr = make_hipPitchedPtr(
            reinterpret_cast<void*>(static_cast<uintptr_t>(dst_handle)), dst_pitch, width_bytes, dst_height ? dst_height : 1
        );
    } else if (dst_memory_type == WP_TEXTURE_MEMORY_ARRAY) {
        copy_params.dstArray = reinterpret_cast<hipArray_t>(static_cast<uintptr_t>(dst_handle));
    } else {
        wp::set_error_string("Invalid destination memory type %d", dst_memory_type);
        return false;
    }
#else
    copy_params.dstMemoryType = static_cast<CUmemorytype>(dst_memory_type);
    if (dst_memory_type == CU_MEMORYTYPE_HOST) {
        copy_params.dstHost = reinterpret_cast<void*>(dst_handle);
    } else if (dst_memory_type == CU_MEMORYTYPE_DEVICE) {
        copy_params.dstDevice = static_cast<CUdeviceptr>(dst_handle);
    } else if (dst_memory_type == CU_MEMORYTYPE_ARRAY) {
        copy_params.dstArray = reinterpret_cast<CUarray>(dst_handle);
    } else {
        wp::set_error_string("Invalid destination memory type %d", dst_memory_type);
        return false;
    }
    copy_params.dstPitch = dst_pitch;
    copy_params.dstHeight = dst_height;
#endif

#if WP_TEXTURE_HIP
    if (src_memory_type == WP_TEXTURE_MEMORY_HOST || src_memory_type == WP_TEXTURE_MEMORY_DEVICE) {
        copy_params.srcPtr = make_hipPitchedPtr(
            reinterpret_cast<void*>(static_cast<uintptr_t>(src_handle)), src_pitch, width_bytes, src_height ? src_height : 1
        );
    } else if (src_memory_type == WP_TEXTURE_MEMORY_ARRAY) {
        copy_params.srcArray = reinterpret_cast<hipArray_t>(static_cast<uintptr_t>(src_handle));
    } else {
        wp::set_error_string("Invalid source memory type %d", src_memory_type);
        return false;
    }
#else
    copy_params.srcMemoryType = static_cast<CUmemorytype>(src_memory_type);
    if (src_memory_type == CU_MEMORYTYPE_HOST) {
        copy_params.srcHost = reinterpret_cast<void*>(src_handle);
    } else if (src_memory_type == CU_MEMORYTYPE_DEVICE) {
        copy_params.srcDevice = static_cast<CUdeviceptr>(src_handle);
    } else if (src_memory_type == CU_MEMORYTYPE_ARRAY) {
        copy_params.srcArray = reinterpret_cast<CUarray>(src_handle);
    } else {
        wp::set_error_string("Invalid source memory type %d", src_memory_type);
        return false;
    }
    copy_params.srcPitch = src_pitch;
    copy_params.srcHeight = src_height;
#endif

#if WP_TEXTURE_HIP
    if (depth <= 1 && dst_is_array && !src_is_array) {
        return check_cuda(hipMemcpy2DToArrayAsync(
            copy_params.dstArray, 0, 0, copy_params.srcPtr.ptr, src_pitch, width_bytes, height ? height : 1,
            copy_params.kind, cuda_stream
        ));
    }
    if (depth <= 1 && src_is_array && !dst_is_array) {
        return check_cuda(hipMemcpy2DFromArrayAsync(
            copy_params.dstPtr.ptr, dst_pitch, copy_params.srcArray, 0, 0, width_bytes, height ? height : 1,
            copy_params.kind, cuda_stream
        ));
    }
    return check_cuda(hipMemcpy3DAsync(&copy_params, cuda_stream));
#else
    return check_cu(cuMemcpy3DAsync_f(&copy_params, cuda_stream));
#endif
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    if (!array_handle || !desc_out) {
        wp::set_error_string("Warp error: NULL array handle");
        return false;
    }

    ContextGuard guard(context);

#if WP_TEXTURE_HIP
    HIP_ARRAY3D_DESCRIPTOR desc;
    if (!check_cuda(hipArray3DGetDescriptor(&desc, reinterpret_cast<hipArray_t>(static_cast<uintptr_t>(array_handle)))))
#else
    CUarray cuda_array = reinterpret_cast<CUarray>(array_handle);
    CUDA_ARRAY3D_DESCRIPTOR desc;
    if (!check_cu(cuArray3DGetDescriptor_f(&desc, cuda_array)))
#endif
        return false;

    switch (desc.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT8;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT16;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT32;
        break;
    case CU_AD_FORMAT_SIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT8;
        break;
    case CU_AD_FORMAT_SIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT16;
        break;
    case CU_AD_FORMAT_SIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT32;
        break;
    case CU_AD_FORMAT_HALF:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT16;
        break;
    case CU_AD_FORMAT_FLOAT:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT32;
        break;
    default:
        wp::set_error_string("Warp error: Unsupported texture format");
        return false;
    }

    desc_out->shape[0] = int32_t(desc.Width);
    desc_out->shape[1] = int32_t(desc.Height);
    desc_out->shape[2] = int32_t(desc.Depth);

    if (desc.Depth == 0) {
        if (desc.Height == 0)
            desc_out->ndim = 1;
        else
            desc_out->ndim = 2;
    } else {
        desc_out->ndim = 3;
    }

    // TODO: desc.Flags?

    desc_out->num_channels = int32_t(desc.NumChannels);

    return true;
}

uint64_t wp_texture_object_create_device(
    void* context,
    uint64_t array_handle,
    int ndim,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    int num_mip_levels
)
{
    if (!array_handle) {
        wp::set_error_string("Null texture array handle");
        return 0;
    }

    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    if (num_mip_levels < 1 || num_mip_levels > WP_TEXTURE_MAX_MIP_LEVELS) {
        wp::set_error_string(
            "Number of texture mip levels must be in [1, %d], got %d", WP_TEXTURE_MAX_MIP_LEVELS, num_mip_levels
        );
        return 0;
    }

    ContextGuard guard(context);

#if WP_TEXTURE_HIP
    hipResourceDesc res_desc = {};
    if (num_mip_levels > 1) {
        res_desc.resType = hipResourceTypeMipmappedArray;
        res_desc.res.mipmap.mipmap = reinterpret_cast<hipMipmappedArray_t>(static_cast<uintptr_t>(array_handle));
    } else {
        res_desc.resType = hipResourceTypeArray;
        res_desc.res.array.array = reinterpret_cast<hipArray_t>(static_cast<uintptr_t>(array_handle));
    }
    hipTextureDesc tex_desc = {};
#else
    CUDA_RESOURCE_DESC res_desc = {};
    if (num_mip_levels > 1) {
        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = reinterpret_cast<CUmipmappedArray>(array_handle);
    } else {
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = reinterpret_cast<CUarray>(array_handle);
    }
    res_desc.flags = 0;

    CUDA_TEXTURE_DESC tex_desc = {};
#endif

    for (int i = 0; i < 3; i++) {
        if (i < ndim)
            tex_desc.addressMode[i] = get_cuda_address_mode(address_modes[i]);
        else
#if WP_TEXTURE_HIP
            tex_desc.addressMode[i] = hipAddressModeClamp;
#else
            tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_CLAMP;
#endif
    }

#if WP_TEXTURE_HIP
    tex_desc.filterMode = (filter_mode == 0) ? hipFilterModePoint : hipFilterModeLinear;
#else
    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
#endif

    // Coordinate mode: normalized [0,1] or texel space [0,width/height]
    // For uint8/uint16 textures, CUDA automatically normalizes values to [0,1] when sampled
    // (since CU_TRSF_READ_AS_INTEGER is NOT set). Float32 textures return values as-is.
#if WP_TEXTURE_HIP
    tex_desc.normalizedCoords = use_normalized_coords;
    tex_desc.readMode = hipReadModeNormalizedFloat;
#else
    tex_desc.flags = use_normalized_coords ? CU_TRSF_NORMALIZED_COORDINATES : 0;
#endif

    tex_desc.maxAnisotropy = 0;
#if WP_TEXTURE_HIP
    tex_desc.mipmapFilterMode = (mip_filter_mode == 0) ? hipFilterModePoint : hipFilterModeLinear;
#else
    tex_desc.mipmapFilterMode = (mip_filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
#endif
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = (num_mip_levels > 1) ? (float)(num_mip_levels - 1) : 0.0f;

#if WP_TEXTURE_HIP
    hipTextureObject_t tex_object = nullptr;
    check_cuda(hipCreateTextureObject(&tex_object, &res_desc, &tex_desc, nullptr));
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tex_object));
#else
    CUtexObject tex_object = 0;
    check_cu(cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr));

    return tex_object;
#endif
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle)
{
    release_or_defer(ReleaseKind::TextureObject, context, tex_handle);
}

uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
    ContextGuard guard(context);

#if WP_TEXTURE_HIP
    hipResourceDesc desc = {};
    desc.resType = hipResourceTypeArray;
    desc.res.array.array = reinterpret_cast<hipArray_t>(static_cast<uintptr_t>(array_handle));
    hipSurfaceObject_t surface = nullptr;
    check_cuda(hipCreateSurfaceObject(&surface, &desc));
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(surface));
#else
    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = reinterpret_cast<cudaArray_t>(array_handle);

    cudaSurfaceObject_t surface = 0;
    check_cuda(cudaCreateSurfaceObject(&surface, &desc));

    return static_cast<uint64_t>(surface);
#endif
}

void wp_surface_object_destroy_device(void* context, uint64_t surface_handle)
{
    release_or_defer(ReleaseKind::SurfaceObject, context, surface_handle);
}

#else  // WP_ENABLE_CUDA

// Stub implementations for non-CUDA builds

uint64_t wp_texture_create_device(
    void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access, int num_mip_levels
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_destroy_device(void* context, uint64_t array_handle, bool is_mipmapped) { }

uint64_t wp_texture_get_mip_level_array_device(void* context, uint64_t mipmap_array_handle, int level)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned width,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

uint64_t wp_texture_object_create_device(
    void* context,
    uint64_t array_handle,
    int ndim,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    int num_mip_levels
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle) { }

WP_API uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

WP_API void wp_surface_object_destroy_device(void* context, uint64_t surface_handle) { }

#endif  // WP_ENABLE_CUDA
