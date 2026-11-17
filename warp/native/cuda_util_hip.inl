// SPDX-License-Identifier: Apache-2.0

#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace wp {
typedef uint32_t GLuint;
typedef uint32_t GLenum;
}

static bool cuda_driver_initialized = false;

bool ContextGuard::always_restore = false;
CudaTimingState* g_cuda_timing_state = nullptr;

bool init_cuda_driver()
{
    if (cuda_driver_initialized)
        return true;
    hipError_t result = hipInit(0);
    if (result != hipSuccess) {
        check_cuda_result(result, __FUNCTION__, __FILE__, __LINE__);
        return false;
    }
    cuda_driver_initialized = true;
    return true;
}

bool is_cuda_driver_initialized() { return cuda_driver_initialized; }

bool check_cuda_result(cudaError_t code, const char* func, const char* file, int line)
{
    if (code == hipSuccess)
        return true;
    wp::set_error_string("Warp HIP error %u: %s (in function %s, %s:%d)", unsigned(code), hipGetErrorString(code), func, file, line);
    return false;
}

bool check_cu_result(CUresult result, const char* func, const char* file, int line)
{
    return check_cuda_result(result, func, file, line);
}

bool get_capture_dependencies(CUstream stream, std::vector<CUgraphNode>& dependencies_ret)
{
    CUstreamCaptureStatus status;
    size_t count = 0;
    const CUgraphNode* dependencies = nullptr;
    dependencies_ret.clear();
    if (!check_cu(cuStreamGetCaptureInfo_f(stream, &status, nullptr, nullptr, &dependencies, &count)))
        return false;
    if (dependencies && count)
        dependencies_ret.assign(dependencies, dependencies + count);
    return true;
}

bool get_graph_leaf_nodes(cudaGraph_t graph, std::vector<cudaGraphNode_t>& leaf_nodes_ret)
{
    if (!graph)
        return false;
    size_t count = 0;
    if (!check_cuda(cudaGraphGetNodes(graph, nullptr, &count)))
        return false;
    std::vector<cudaGraphNode_t> nodes(count);
    if (!check_cuda(cudaGraphGetNodes(graph, nodes.data(), &count)))
        return false;
    leaf_nodes_ret.clear();
    for (cudaGraphNode_t node : nodes) {
        size_t dependent_count = 0;
        if (!check_cu(cuGraphNodeGetDependentNodes_f(node, nullptr, &dependent_count)))
            return false;
        if (dependent_count == 0)
            leaf_nodes_ret.push_back(node);
    }
    return true;
}

bool get_dependent_leaf_nodes(cudaGraphNode_t, std::vector<cudaGraphNode_t>& leaf_nodes_ret)
{
    leaf_nodes_ret.clear();
    return false;
}

NodeDependencyResult graph_node_depends_on(cudaGraphNode_t, cudaGraphNode_t)
{
    return NODE_DEPENDENCY_RESULT_ERROR;
}

GraphAllocQueryResult graph_alloc_query(cudaGraphNode_t, cudaGraphNode_t)
{
    return GRAPH_ALLOC_QUERY_RESULT_ERROR;
}

CUresult cuDriverGetVersion_f(int* version) { return hipDriverGetVersion(version); }

CUresult cuGetErrorName_f(CUresult result, const char** pstr)
{
    if (!pstr)
        return hipErrorInvalidValue;
    *pstr = hipGetErrorName(result);
    return *pstr ? hipSuccess : hipErrorInvalidValue;
}

CUresult cuGetErrorString_f(CUresult result, const char** pstr)
{
    if (!pstr)
        return hipErrorInvalidValue;
    *pstr = hipGetErrorString(result);
    return *pstr ? hipSuccess : hipErrorInvalidValue;
}

CUresult cuInit_f(unsigned int flags) { return hipInit(flags); }
CUresult cuDeviceGet_f(CUdevice* dev, int ordinal) { return hipDeviceGet(dev, ordinal); }
CUresult cuDeviceGetCount_f(int* count) { return hipGetDeviceCount(count); }
CUresult cuDeviceGetName_f(char* name, int len, CUdevice dev) { return hipDeviceGetName(name, len, dev); }
CUresult cuDeviceGetAttribute_f(int* value, CUdevice_attribute attrib, CUdevice dev)
{
    return hipDeviceGetAttribute(value, attrib, dev);
}
CUresult cuDeviceGetUuid_f(CUuuid* uuid, CUdevice dev) { return hipDeviceGetUuid(uuid, dev); }
CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev) { return hipDevicePrimaryCtxRetain(ctx, dev); }
CUresult cuDevicePrimaryCtxRelease_f(CUdevice dev) { return hipDevicePrimaryCtxRelease(dev); }
CUresult cuDeviceCanAccessPeer_f(int* access, CUdevice dev, CUdevice peer) { return hipDeviceCanAccessPeer(access, dev, peer); }
CUresult cuMemGetInfo_f(size_t* free, size_t* total) { return hipMemGetInfo(free, total); }

CUresult cuCtxGetCurrent_f(CUcontext* ctx) { return hipCtxGetCurrent(ctx); }
CUresult cuCtxSetCurrent_f(CUcontext ctx) { return hipCtxSetCurrent(ctx); }
CUresult cuCtxPushCurrent_f(CUcontext ctx) { return hipCtxPushCurrent(ctx); }
CUresult cuCtxPopCurrent_f(CUcontext* ctx) { return hipCtxPopCurrent(ctx); }
CUresult cuCtxSynchronize_f() { return hipDeviceSynchronize(); }
CUresult cuProfilerStart_f() { return hipProfilerStart(); }
CUresult cuProfilerStop_f() { return hipProfilerStop(); }
CUresult cuCtxGetDevice_f(CUdevice* dev) { return hipCtxGetDevice(dev); }
CUresult cuCtxCreate_f(CUcontext* ctx, unsigned int flags, CUdevice dev) { return hipCtxCreate(ctx, flags, dev); }
CUresult cuCtxDestroy_f(CUcontext ctx) { return hipCtxDestroy(ctx); }

CUresult cuCtxEnablePeerAccess_f(CUcontext peer_ctx, unsigned int flags)
{
    if (!peer_ctx)
        return hipErrorInvalidValue;
    CUdevice dev = -1;
    CUcontext previous = nullptr;
    CUresult result = hipCtxGetCurrent(&previous);
    if (result != hipSuccess)
        return result;
    result = hipCtxSetCurrent(peer_ctx);
    if (result == hipSuccess)
        result = hipCtxGetDevice(&dev);
    CUresult restore_result = hipCtxSetCurrent(previous);
    if (result == hipSuccess)
        result = restore_result;
    if (result != hipSuccess)
        return result;
    return hipDeviceEnablePeerAccess(dev, flags);
}

CUresult cuCtxDisablePeerAccess_f(CUcontext peer_ctx)
{
    if (!peer_ctx)
        return hipErrorInvalidValue;
    CUdevice dev = -1;
    CUcontext previous = nullptr;
    CUresult result = hipCtxGetCurrent(&previous);
    if (result != hipSuccess)
        return result;
    result = hipCtxSetCurrent(peer_ctx);
    if (result == hipSuccess)
        result = hipCtxGetDevice(&dev);
    CUresult restore_result = hipCtxSetCurrent(previous);
    if (result == hipSuccess)
        result = restore_result;
    if (result != hipSuccess)
        return result;
    return hipDeviceDisablePeerAccess(dev);
}

CUresult cuStreamCreate_f(CUstream* stream, unsigned int flags) { return hipStreamCreateWithFlags(stream, flags); }
CUresult cuStreamDestroy_f(CUstream stream) { return hipStreamDestroy(stream); }
CUresult cuStreamQuery_f(CUstream stream) { return hipStreamQuery(stream); }
CUresult cuStreamSynchronize_f(CUstream stream) { return hipStreamSynchronize(stream); }
CUresult cuStreamWaitEvent_f(CUstream stream, CUevent event, unsigned int flags) { return hipStreamWaitEvent(stream, event, flags); }
CUresult cuStreamGetCtx_f(CUstream stream, CUcontext* context) { return hipStreamGetCtx(stream, context); }

CUresult cuStreamGetCaptureInfo_f(CUstream stream, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph,
                                  const CUgraphNode** dependencies, size_t* count)
{
    unsigned long long capture_id = 0;
    CUresult result = hipStreamGetCaptureInfo_v2(stream, status, id ? &capture_id : nullptr, graph, dependencies, count);
    if (result == hipSuccess && id)
        *id = static_cast<cuuint64_t>(capture_id);
    return result;
}

CUresult cuStreamUpdateCaptureDependencies_f(CUstream stream, CUgraphNode* nodes, size_t count, unsigned int flags)
{
    return hipStreamUpdateCaptureDependencies(stream, nodes, count, flags);
}

CUresult cuStreamCreateWithPriority_f(CUstream* stream, unsigned int flags, int priority)
{
    return hipStreamCreateWithPriority(stream, flags, priority);
}
CUresult cuStreamGetPriority_f(CUstream stream, int* priority) { return hipStreamGetPriority(stream, priority); }
CUresult cuEventCreate_f(CUevent* event, unsigned int flags) { return hipEventCreateWithFlags(event, flags); }
CUresult cuEventDestroy_f(CUevent event) { return hipEventDestroy(event); }
CUresult cuEventQuery_f(CUevent event) { return hipEventQuery(event); }
CUresult cuEventRecord_f(CUevent event, CUstream stream) { return hipEventRecord(event, stream); }
CUresult cuEventRecordWithFlags_f(CUevent event, CUstream stream, unsigned int flags)
{
    return hipEventRecordWithFlags(event, stream, flags);
}
CUresult cuEventSynchronize_f(CUevent event) { return hipEventSynchronize(event); }
CUresult cuGraphNodeGetDependentNodes_f(CUgraphNode node, CUgraphNode* nodes, size_t* count)
{
    return hipGraphNodeGetDependentNodes(node, nodes, count);
}
CUresult cuGraphNodeGetType_f(CUgraphNode node, CUgraphNodeType* type) { return hipGraphNodeGetType(node, type); }

CUresult cuModuleUnload_f(CUmodule module) { return hipModuleUnload(static_cast<hipModule_t>(module)); }
CUresult cuModuleGetFunction_f(CUfunction* function, CUmodule module, const char* name)
{
    hipFunction_t hip_function = nullptr;
    CUresult result = hipModuleGetFunction(&hip_function, static_cast<hipModule_t>(module), name);
    if (result == hipSuccess)
        *function = static_cast<void*>(hip_function);
    return result;
}

CUresult cuLaunchKernel_f(CUfunction function, unsigned int gx, unsigned int gy, unsigned int gz,
                          unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared_bytes,
                          CUstream stream, void** params, void** extra)
{
    (void)extra;
    return hipModuleLaunchKernel(static_cast<hipFunction_t>(function), gx, gy, gz, bx, by, bz,
                                 shared_bytes, stream, params, nullptr);
}

CUresult cuOccupancyMaxPotentialBlockSize_f(int* min_grid, int* block_size, CUfunction function,
                                            CUoccupancyB2DSize callback, size_t dynamic_shared, int block_limit)
{
    (void)callback;
    int grid_size = 0;
    return hipModuleOccupancyMaxPotentialBlockSize(&grid_size, block_size,
        static_cast<hipFunction_t>(function), dynamic_shared, block_limit) == hipSuccess
        ? (*min_grid = grid_size, hipSuccess) : hipErrorInvalidValue;
}

CUresult cuMemcpyPeerAsync_f(CUdeviceptr dst, CUcontext dst_ctx, CUdeviceptr src, CUcontext src_ctx,
                             size_t size, CUstream stream)
{
    CUcontext previous = nullptr;
    CUdevice dst_dev = -1, src_dev = -1;
    CUresult result = hipCtxGetCurrent(&previous);
    if (result != hipSuccess)
        return result;
    result = hipCtxSetCurrent(dst_ctx);
    if (result == hipSuccess)
        result = hipCtxGetDevice(&dst_dev);
    if (result == hipSuccess)
        result = hipCtxSetCurrent(src_ctx);
    if (result == hipSuccess)
        result = hipCtxGetDevice(&src_dev);
    CUresult restore_result = hipCtxSetCurrent(previous);
    if (result == hipSuccess)
        result = restore_result;
    if (result != hipSuccess)
        return result;
    return hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, size, stream);
}

CUresult cuPointerGetAttribute_f(void* data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    return hipPointerGetAttribute(data, attribute, ptr);
}

#define WP_HIP_UNSUPPORTED(name, signature) CUresult name signature { return hipErrorNotSupported; }
WP_HIP_UNSUPPORTED(cuGraphicsMapResources_f, (unsigned int, CUgraphicsResource*, CUstream))
WP_HIP_UNSUPPORTED(cuGraphicsUnmapResources_f, (unsigned int, CUgraphicsResource*, CUstream))
WP_HIP_UNSUPPORTED(cuGraphicsResourceGetMappedPointer_f, (CUdeviceptr*, size_t*, CUgraphicsResource))
WP_HIP_UNSUPPORTED(cuGraphicsGLRegisterBuffer_f, (CUgraphicsResource*, unsigned int, unsigned int))
WP_HIP_UNSUPPORTED(cuGraphicsGLRegisterImage_f, (CUgraphicsResource*, unsigned int, unsigned int, unsigned int))
WP_HIP_UNSUPPORTED(cuGraphicsSubResourceGetMappedArray_f, (CUarray*, CUgraphicsResource, unsigned int, unsigned int))
WP_HIP_UNSUPPORTED(cuGraphicsUnregisterResource_f, (CUgraphicsResource))
WP_HIP_UNSUPPORTED(cuModuleGetGlobal_f, (CUdeviceptr*, size_t*, CUmodule, const char*))
CUresult cuFuncSetAttribute_f(CUfunction function, CUfunction_attribute attribute, int value)
{
    if (attribute != HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
        return hipErrorNotSupported;
    return hipFuncSetAttribute(reinterpret_cast<const void*>(function), hipFuncAttributeMaxDynamicSharedMemorySize, value);
}
CUresult cuFuncGetAttribute_f(int* value, CUfunction_attribute attribute, CUfunction function)
{
    return hipFuncGetAttribute(value, attribute, reinterpret_cast<hipFunction_t>(function));
}
WP_HIP_UNSUPPORTED(cuIpcGetEventHandle_f, (CUipcEventHandle*, CUevent))
WP_HIP_UNSUPPORTED(cuIpcOpenEventHandle_f, (CUevent*, CUipcEventHandle))
WP_HIP_UNSUPPORTED(cuIpcGetMemHandle_f, (CUipcMemHandle*, CUdeviceptr))
WP_HIP_UNSUPPORTED(cuIpcOpenMemHandle_f, (CUdeviceptr*, CUipcMemHandle, unsigned int))
WP_HIP_UNSUPPORTED(cuIpcCloseMemHandle_f, (CUdeviceptr))
WP_HIP_UNSUPPORTED(cuArrayCreate_f, (CUarray*, const CUDA_ARRAY_DESCRIPTOR*))
CUresult cuArrayDestroy_f(CUarray array) { return hipArrayDestroy(array); }
CUresult cuArray3DCreate_f(CUarray* array, const CUDA_ARRAY3D_DESCRIPTOR* desc)
{
    return hipArray3DCreate(array, desc);
}
CUresult cuArray3DGetDescriptor_f(CUDA_ARRAY3D_DESCRIPTOR* desc, CUarray array)
{
    return hipArray3DGetDescriptor(desc, array);
}
WP_HIP_UNSUPPORTED(cuMemcpy2D_f, (const CUDA_MEMCPY2D*))
WP_HIP_UNSUPPORTED(cuMemcpy2DAsync_f, (const CUDA_MEMCPY2D*, CUstream))
WP_HIP_UNSUPPORTED(cuMemcpy3D_f, (const CUDA_MEMCPY3D*))
WP_HIP_UNSUPPORTED(cuMemcpy3DAsync_f, (const CUDA_MEMCPY3D*, CUstream))
WP_HIP_UNSUPPORTED(cuTexObjectCreate_f, (CUtexObject*, const CUDA_RESOURCE_DESC*, const CUDA_TEXTURE_DESC*, const CUDA_RESOURCE_VIEW_DESC*))
CUresult cuTexObjectDestroy_f(CUtexObject object) { return hipDestroyTextureObject(object); }
CUresult cuMipmappedArrayCreate_f(CUmipmappedArray* array, const CUDA_ARRAY3D_DESCRIPTOR* desc, unsigned int levels)
{
    CUDA_ARRAY3D_DESCRIPTOR mutable_desc = *desc;
    return hipMipmappedArrayCreate(array, &mutable_desc, levels);
}
CUresult cuMipmappedArrayDestroy_f(CUmipmappedArray array) { return hipMipmappedArrayDestroy(array); }
CUresult cuMipmappedArrayGetLevel_f(CUarray* array, CUmipmappedArray mipmap, unsigned int level)
{
    return hipMipmappedArrayGetLevel(array, mipmap, level);
}
#undef WP_HIP_UNSUPPORTED

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
