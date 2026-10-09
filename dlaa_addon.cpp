// Arkham Knight DLAA - ReShade Addon (DX11) + NVIDIA NGX
// Clean camera motion vector generation & unjittered reprojection

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <reshade.hpp>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_params.h>
#include <nvsdk_ngx_helpers.h>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

using namespace DirectX;
using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "AK DLAA";
extern "C" __declspec(dllexport) const char *DESCRIPTION = "Clean DLAA for Batman: Arkham Knight";

// ------------------------------------------------------------ helpers ----
static std::string ExeDir() {
    char p[MAX_PATH];
    GetModuleFileNameA(nullptr, p, MAX_PATH);
    std::string s(p);
    return s.substr(0, s.find_last_of("\\/") + 1);
}

static void Log(const char *fmt, ...) {
    static FILE *f = nullptr;
    if (!f) f = fopen((ExeDir() + "dlaa_log.txt").c_str(), "a");
    if (!f) return;
    va_list a;
    va_start(a, fmt);
    vfprintf(f, fmt, a);
    va_end(a);
    fputc('\n', f);
    fflush(f);
}

template <class T> static void Rel(T *&p) { if (p) { p->Release(); p = nullptr; } }

// -------------------------------------------------------------- config ----
struct Config {
    int   triggerIndex = 2;
    int   logFrame = 300;
    bool  hdrColor = true;
    bool  applyJitter = true;
    bool  swapMatrices = false;
    float jitterSignX = 1.0f;
    float jitterSignY = 1.0f;
    int   colorFormat = 26; // DXGI_FORMAT_R11G11B10_FLOAT
    int   phases = 16;
    float sharpness = 0.35f;
    float blend = 0.0f;
    int   preset = 11;      // Preset K
} cfg;

static void LoadConfig() {
    std::string ini = ExeDir() + "dlaa.ini";
    auto I = [&](const char *k, int d) { return (int)GetPrivateProfileIntA("dlaa", k, d, ini.c_str()); };
    auto F = [&](const char *k, float d) {
        char def[32], buf[32];
        snprintf(def, 32, "%f", d);
        GetPrivateProfileStringA("dlaa", k, def, buf, 32, ini.c_str());
        return (float)atof(buf);
    };
    cfg.triggerIndex = I("TriggerIndex", 2);
    cfg.logFrame     = I("LogFrame", 300);
    cfg.hdrColor     = I("HDRColor", 1) != 0;
    cfg.applyJitter  = I("ApplyJitter", 1) != 0;
    cfg.swapMatrices = I("SwapMatrices", 0) != 0;
    cfg.jitterSignX  = F("JitterSignX", 1.0f);
    cfg.jitterSignY  = F("JitterSignY", 1.0f);
    cfg.sharpness    = F("Sharpness", 0.35f);
    cfg.blend        = F("OriginalBlend", 0.0f);
    cfg.preset       = I("Preset", 11);
}

// --------------------------------------------------------------- state ----
static ID3D11Device *g_dev = nullptr;
static NVSDK_NGX_Parameter *g_ngx = nullptr;
static NVSDK_NGX_Handle *g_dlss = nullptr;
static uint32_t g_w = 0, g_h = 0;
static ID3D11Texture2D *g_inTex = nullptr, *g_outTex = nullptr, *g_mvTex = nullptr;
static ID3D11UnorderedAccessView *g_mvUAV = nullptr;
static ID3D11ComputeShader *g_mvCS = nullptr;
static ID3D11Buffer *g_mvCB = nullptr;
static ID3D11Texture2D *g_sharpTex = nullptr;
static ID3D11UnorderedAccessView *g_sharpUAV = nullptr;
static ID3D11ShaderResourceView *g_outSRV = nullptr, *g_inSRV = nullptr;
static ID3D11ComputeShader *g_postCS = nullptr;
static ID3D11Buffer *g_postCB = nullptr;
static ID3D11Texture2D *g_depthTex = nullptr;
static ID3D11ShaderResourceView *g_depthSRV = nullptr;
static uint32_t g_bbW = 0, g_bbH = 0;
static uint32_t g_frame = 0, g_dispatchCount = 0, g_matchCount = 0;
static bool g_done = false, g_reset = true, g_haveView = false, g_failed = false, g_inside = false;
static bool g_enabled = true;
static uint32_t g_applyCount = 0;

// Matrices (Column-major storage matching UE3)
static XMFLOAT4X4 g_curVP;
static XMFLOAT4X4 g_prevVP;
static XMFLOAT4X4 g_invCurVP;

static float g_near = 10.0f;
static float g_jx = 0.0f, g_jy = 0.0f;
static std::mutex g_mtx;
static std::unordered_map<uint64_t, void *> g_mapped;

static float Halton(uint32_t i, uint32_t b) {
    float f = 1.0f, r = 0.0f;
    while (i > 0) { f /= b; r += f * (i % b); i /= b; }
    return r;
}

// ------------------------------------------------- view constants capture ----
static bool LooksLikeView(const float *c, size_t bytes) {
    if (bytes < 23 * 16) return false;
    const float *v8 = c + 8 * 4, *v10 = c + 10 * 4, *v18 = c + 18 * 4, *v22 = c + 22 * 4;
    if (!(v8[0] == 0.0f && v8[1] == 0.0f && v8[2] == 0.0f && v8[3] == 1.0f)) return false;
    if (!(v10[3] == 1.0f && v18[3] == 1.0f)) return false;
    if (!(v22[0] >= 320.0f && v22[0] <= 16384.0f && v22[1] >= 240.0f && v22[1] <= 16384.0f)) return false;
    if (fabsf(v22[0] * v22[2] - 1.0f) > 0.01f || fabsf(v22[1] * v22[3] - 1.0f) > 0.01f) return false;
    if (g_bbW && ((uint32_t)v22[0] != g_bbW || (uint32_t)v22[1] != g_bbH)) return false;
    return true;
}

static void ProcessViewConstants(float *c, size_t bytes) {
    if (!LooksLikeView(c, bytes)) return;
    std::lock_guard<std::mutex> lock(g_mtx);

    int curI = cfg.swapMatrices ? 4 : 0;
    int prevI = cfg.swapMatrices ? 0 : 4;

    // Load matrices without any jitter
    XMMATRIX curMat = XMLoadFloat4x4(reinterpret_cast<const XMFLOAT4X4 *>(c + curI * 4));
    XMMATRIX prevMat = XMLoadFloat4x4(reinterpret_cast<const XMFLOAT4X4 *>(c + prevI * 4));

    XMVECTOR det;
    XMMATRIX invCurMat = XMMatrixInverse(&det, curMat);

    XMStoreFloat4x4(&g_curVP, curMat);
    XMStoreFloat4x4(&g_prevVP, prevMat);
    XMStoreFloat4x4(&g_invCurVP, invCurMat);

    float nz = fabsf(c[20 * 4]);
    if (nz > 0.01f && nz < 10000.0f) g_near = nz;
    g_haveView = true;

    // Apply subpixel camera jitter only to the buffer being sent to rendering
    if (cfg.applyJitter && g_enabled && g_bbW) {
        ++g_applyCount;
        float jx = 2.0f * g_jx / (float)g_bbW;
        float jy = -2.0f * g_jy / (float)g_bbH;

        for (int k = 0; k < 4; ++k) {
            float *col = c + (curI + k) * 4;
            col[0] += jx * col[3];
            col[1] += jy * col[3];
        }
    }
}

static void on_map_buffer_region(device *, resource res, uint64_t offset, uint64_t, map_access, void **data) {
    if (g_inside || offset != 0 || !data || !*data) return;
    std::lock_guard<std::mutex> lock(g_mtx);
    g_mapped[res.handle] = *data;
}

static void on_unmap_buffer_region(device *dev, resource res) {
    void *p = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        auto it = g_mapped.find(res.handle);
        if (it == g_mapped.end()) return;
        p = it->second;
        g_mapped.erase(it);
    }
    resource_desc d = dev->get_resource_desc(res);
    if (d.type == resource_type::buffer && d.buffer.size >= 368 && d.buffer.size <= 4096)
        ProcessViewConstants(static_cast<float *>(p), (size_t)d.buffer.size);
}

static bool on_update_buffer_region(device *, const void *data, resource, uint64_t offset, uint64_t size) {
    if (!g_inside && offset == 0 && size >= 368 && size <= 4096)
        ProcessViewConstants(const_cast<float *>(static_cast<const float *>(data)), (size_t)size);
    return false;
}

// --------------------------------------------------------------- shaders ----
struct MVCB {
    XMFLOAT4X4 InvCurViewProj;
    XMFLOAT4X4 PrevViewProj;
    float params[4]; // x=width, y=height, z=near, w=unused
};

static const char *kMVShader = R"HLSL(
cbuffer CB : register(b0) {
    float4x4 InvCurViewProj;
    float4x4 PrevViewProj;
    float4 prm;
};
Texture2D<float> Depth : register(t0);
RWTexture2D<float2> MV : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)prm.x || id.y >= (uint)prm.y) return;
    
    float depth = Depth.Load(int3(id.xy, 0));
    float2 uv = (float2(id.xy) + 0.5) / prm.xy;
    
    // Arkham Knight Reversed-Z NDC [-1..1, 1..-1, depth..1]
    float4 clipCurr = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    
    // Unproject to camera-relative world space
    float4 worldPos = mul(clipCurr, InvCurViewProj);
    worldPos /= worldPos.w;
    
    // Reproject to previous frame's clip space
    float4 clipPrev = mul(worldPos, PrevViewProj);
    
    float2 mv = float2(0.0, 0.0);
    if (clipPrev.w > 1e-5) {
        float2 prevNDC = clipPrev.xy / clipPrev.w;
        float2 prevUV = float2(prevNDC.x * 0.5 + 0.5, 0.5 - prevNDC.y * 0.5);
        
        // NVIDIA NGX convention: Current - Previous in normalized UV space
        mv = uv - prevUV;
    }
    MV[id.xy] = mv;
}
)HLSL";

static const char *kPostShader = R"HLSL(
cbuffer CB : register(b0) { float4 prm; }; // x=sharpness, y=blend, z=width, w=height
Texture2D<float4> Dlaa : register(t0);
Texture2D<float4> Orig : register(t1);
RWTexture2D<float4> Dst : register(u0);

float3 Enc(float3 c) { return sqrt(max(c, 0.0)); }
float3 Dec(float3 c) { return c * c; }

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    int2 sz = int2((int)prm.z, (int)prm.w);
    if (id.x >= (uint)sz.x || id.y >= (uint)sz.y) return;
    int2 p = int2(id.xy);
    int2 lo = int2(0, 0);
    int2 hi = sz - 1;
    float4 c4 = Dlaa.Load(int3(p, 0));
    float3 c = Enc(c4.rgb);
    float3 n = Enc(Dlaa.Load(int3(clamp(p + int2(0, -1), lo, hi), 0)).rgb);
    float3 s = Enc(Dlaa.Load(int3(clamp(p + int2(0, 1), lo, hi), 0)).rgb);
    float3 w = Enc(Dlaa.Load(int3(clamp(p + int2(-1, 0), lo, hi), 0)).rgb);
    float3 e = Enc(Dlaa.Load(int3(clamp(p + int2(1, 0), lo, hi), 0)).rgb);
    float3 o = Enc(Orig.Load(int3(p, 0)).rgb);

    float3 blur = (n + s + w + e) * 0.25;
    float3 base = lerp(c, o, prm.y);
    float3 mn = min(base, min(c, min(min(n, s), min(w, e))));
    float3 mx = max(base, max(c, max(max(n, s), max(w, e))));
    float3 outc = clamp(base + prm.x * (c - blur), mn, mx);
    Dst[id.xy] = float4(Dec(outc), c4.a);
}
)HLSL";

// ------------------------------------------------------------------ NGX ----
static bool InitNGX(ID3D11Device *dev) {
    if (NVSDK_NGX_FAILED(NVSDK_NGX_D3D11_Init_with_ProjectID(
            "a0f57b54-1daf-4934-90ae-c4c0a0e1e7a1", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", L".", dev)))
        return false;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_D3D11_GetCapabilityParameters(&g_ngx))) return false;
    int ok = 0;
    g_ngx->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &ok);
    return ok != 0;
}

static bool CreateDLSS(ID3D11DeviceContext *ctx, uint32_t w, uint32_t h) {
    if (g_dlss) { NVSDK_NGX_D3D11_ReleaseFeature(g_dlss); g_dlss = nullptr; }
    NVSDK_NGX_DLSS_Create_Params p = {};
    p.Feature.InWidth = p.Feature.InTargetWidth = w;
    p.Feature.InHeight = p.Feature.InTargetHeight = h;
    p.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    p.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_AutoExposure | NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    if (cfg.hdrColor) p.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;

    g_ngx->Set("DLSS.Hint.Render.Preset.DLAA", (unsigned int)cfg.preset);
    g_ngx->Set("DLSS.Hint.Render.Preset.Quality", (unsigned int)cfg.preset);
    g_ngx->Set("DLSS.Hint.Render.Preset.UltraQuality", (unsigned int)cfg.preset);

    if (NVSDK_NGX_FAILED(NGX_D3D11_CREATE_DLSS_EXT(ctx, &g_dlss, g_ngx, &p))) return false;
    g_reset = true;
    return true;
}

static bool EnsureResources(ID3D11DeviceContext *ctx, const D3D11_TEXTURE2D_DESC &sd) {
    if (g_dlss && g_w == sd.Width && g_h == sd.Height && g_inTex) return true;
    Rel(g_inTex); Rel(g_outTex); Rel(g_mvTex); Rel(g_mvUAV);
    Rel(g_sharpTex); Rel(g_sharpUAV); Rel(g_outSRV); Rel(g_inSRV);
    g_w = sd.Width; g_h = sd.Height;

    D3D11_TEXTURE2D_DESC d = sd;
    d.MipLevels = 1; d.ArraySize = 1; d.SampleDesc.Count = 1; d.SampleDesc.Quality = 0;
    d.Usage = D3D11_USAGE_DEFAULT; d.CPUAccessFlags = 0; d.MiscFlags = 0;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(g_dev->CreateTexture2D(&d, nullptr, &g_inTex))) return false;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(g_dev->CreateTexture2D(&d, nullptr, &g_outTex))) return false;
    if (FAILED(g_dev->CreateTexture2D(&d, nullptr, &g_sharpTex))) return false;
    if (FAILED(g_dev->CreateUnorderedAccessView(g_sharpTex, nullptr, &g_sharpUAV))) return false;
    if (FAILED(g_dev->CreateShaderResourceView(g_outTex, nullptr, &g_outSRV))) return false;
    if (FAILED(g_dev->CreateShaderResourceView(g_inTex, nullptr, &g_inSRV))) return false;
    d.Format = DXGI_FORMAT_R16G16_FLOAT;
    if (FAILED(g_dev->CreateTexture2D(&d, nullptr, &g_mvTex))) return false;
    if (FAILED(g_dev->CreateUnorderedAccessView(g_mvTex, nullptr, &g_mvUAV))) return false;

    if (!g_mvCS) {
        ID3DBlob *blob = nullptr, *err = nullptr;
        HRESULT hr = D3DCompile(kMVShader, strlen(kMVShader), "mv", nullptr, nullptr, "main", "cs_5_0", 0, 0, &blob, &err);
        if (FAILED(hr)) {
            Log("MV shader compile failed: %s", err ? (const char *)err->GetBufferPointer() : "unknown");
            Rel(err);
            return false;
        }
        hr = g_dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &g_mvCS);
        Rel(blob); Rel(err);
        if (FAILED(hr)) return false;
    }
    if (!g_mvCB) {
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = sizeof(MVCB);
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(g_dev->CreateBuffer(&bd, nullptr, &g_mvCB))) return false;
    }
    if (!g_postCS) {
        ID3DBlob *blob = nullptr, *err = nullptr;
        HRESULT hr = D3DCompile(kPostShader, strlen(kPostShader), "post", nullptr, nullptr, "main", "cs_5_0", 0, 0, &blob, &err);
        if (FAILED(hr)) {
            Log("post shader compile failed: %s", err ? (const char *)err->GetBufferPointer() : "unknown");
            Rel(err);
            return false;
        }
        hr = g_dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &g_postCS);
        Rel(blob); Rel(err);
        if (FAILED(hr)) return false;
    }
    if (!g_postCB) {
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = 16;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(g_dev->CreateBuffer(&bd, nullptr, &g_postCB))) return false;
    }
    return CreateDLSS(ctx, g_w, g_h);
}

static bool EnsureDepthSRV() {
    if (g_depthSRV) return true;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
    sv.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sv.Texture2D.MipLevels = 1;
    return SUCCEEDED(g_dev->CreateShaderResourceView(g_depthTex, &sv, &g_depthSRV));
}

static void SetDepth(ID3D11Texture2D *t) {
    if (t == g_depthTex) return;
    Rel(g_depthSRV);
    Rel(g_depthTex);
    t->AddRef();
    g_depthTex = t;
}

struct StateBlock {
    ID3D11ComputeShader *cs = nullptr;
    ID3D11ShaderResourceView *srv[16] = {};
    ID3D11UnorderedAccessView *uav[8] = {};
    ID3D11Buffer *cb[14] = {};
    ID3D11SamplerState *smp[16] = {};
    ID3D11RenderTargetView *rtv[8] = {};
    ID3D11DepthStencilView *dsv = nullptr;

    void Save(ID3D11DeviceContext *c) {
        c->CSGetShader(&cs, nullptr, nullptr);
        c->CSGetShaderResources(0, 16, srv);
        c->CSGetUnorderedAccessViews(0, 8, uav);
        c->CSGetConstantBuffers(0, 14, cb);
        c->CSGetSamplers(0, 16, smp);
        c->OMGetRenderTargets(8, rtv, &dsv);
    }
    void Restore(ID3D11DeviceContext *c) {
        UINT keep[8];
        for (auto &k : keep) k = 0xFFFFFFFFu;
        c->CSSetShader(cs, nullptr, 0);
        c->CSSetShaderResources(0, 16, srv);
        c->CSSetUnorderedAccessViews(0, 8, uav, keep);
        c->CSSetConstantBuffers(0, 14, cb);
        c->CSSetSamplers(0, 16, smp);
        c->OMSetRenderTargets(8, rtv, dsv);
        Rel(cs);
        for (auto &p : srv) Rel(p);
        for (auto &p : uav) Rel(p);
        for (auto &p : cb) Rel(p);
        for (auto &p : smp) Rel(p);
        for (auto &p : rtv) Rel(p);
        Rel(dsv);
    }
};

static void RunDLAA(ID3D11DeviceContext *ctx, ID3D11Texture2D *scene) {
    if (g_failed || !g_depthTex) return;
    D3D11_TEXTURE2D_DESC sd, dd;
    scene->GetDesc(&sd);
    g_depthTex->GetDesc(&dd);
    if (!(dd.BindFlags & D3D11_BIND_SHADER_RESOURCE) || dd.Width != sd.Width || dd.Height != sd.Height) {
        Log("depth texture unusable (bind flags 0x%x, %ux%u)", dd.BindFlags, dd.Width, dd.Height);
        g_failed = true;
        return;
    }
    if (!EnsureResources(ctx, sd) || !EnsureDepthSRV()) {
        Log("resource setup failed");
        g_failed = true;
        return;
    }

    StateBlock st;
    st.Save(ctx);
    ctx->OMSetRenderTargets(0, nullptr, nullptr);

    // 1) Compute unjittered camera motion vectors
    MVCB cb;
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        cb.InvCurViewProj = g_invCurVP;
        cb.PrevViewProj   = g_prevVP;
    }
    cb.params[0] = (float)g_w;
    cb.params[1] = (float)g_h;
    cb.params[2] = g_near;
    cb.params[3] = 0.0f;

    ctx->UpdateSubresource(g_mvCB, 0, nullptr, &cb, 0, 0);
    ctx->CSSetShader(g_mvCS, nullptr, 0);
    ctx->CSSetConstantBuffers(0, 1, &g_mvCB);
    ctx->CSSetShaderResources(0, 1, &g_depthSRV);
    UINT ic = 0xFFFFFFFFu;
    ctx->CSSetUnorderedAccessViews(0, 1, &g_mvUAV, &ic);
    ctx->Dispatch((g_w + 7) / 8, (g_h + 7) / 8, 1);

    ID3D11UnorderedAccessView *nu = nullptr;
    ctx->CSSetUnorderedAccessViews(0, 1, &nu, &ic);
    ID3D11ShaderResourceView *ns = nullptr;
    ctx->CSSetShaderResources(0, 1, &ns);

    // 2) DLSS Evaluation
    ctx->CopyResource(g_inTex, scene);
    NVSDK_NGX_D3D11_DLSS_Eval_Params e = {};
    e.Feature.pInColor = g_inTex;
    e.Feature.pInOutput = g_outTex;
    e.pInDepth = g_depthTex;
    e.pInMotionVectors = g_mvTex;
    e.InJitterOffsetX = cfg.jitterSignX * g_jx;
    e.InJitterOffsetY = cfg.jitterSignY * g_jy;
    e.InRenderSubrectDimensions = { g_w, g_h };
    e.InReset = g_reset ? 1 : 0;
    e.InMVScaleX = 1.0f;
    e.InMVScaleY = 1.0f;
    e.InPreExposure = 1.0f;
    e.Feature.InSharpness = 0.0f;

    if (NVSDK_NGX_SUCCEED(NGX_D3D11_EVALUATE_DLSS_EXT(ctx, g_dlss, g_ngx, &e))) {
        if (cfg.sharpness > 0.001f || cfg.blend > 0.001f) {
            float pc[4] = { cfg.sharpness, cfg.blend, (float)g_w, (float)g_h };
            ctx->UpdateSubresource(g_postCB, 0, nullptr, pc, 0, 0);
            ctx->CSSetShader(g_postCS, nullptr, 0);
            ctx->CSSetConstantBuffers(0, 1, &g_postCB);
            ID3D11ShaderResourceView *sv[2] = { g_outSRV, g_inSRV };
            ctx->CSSetShaderResources(0, 2, sv);
            ctx->CSSetUnorderedAccessViews(0, 1, &g_sharpUAV, &ic);
            ctx->Dispatch((g_w + 7) / 8, (g_h + 7) / 8, 1);
            ctx->CSSetUnorderedAccessViews(0, 1, &nu, &ic);
            ID3D11ShaderResourceView *none[2] = { nullptr, nullptr };
            ctx->CSSetShaderResources(0, 2, none);
            ctx->CopyResource(scene, g_sharpTex);
        } else {
            ctx->CopyResource(scene, g_outTex);
        }
        if (g_reset) Log("DLAA running stably at frame %u, dispatch %u", g_frame, g_dispatchCount);
        g_reset = false;
    } else {
        Log("DLSS evaluate failed at frame %u", g_frame);
    }
    st.Restore(ctx);
    g_done = true;
}

// ---------------------------------------------------------------- events ----
static void on_init_device(device *d) {
    LoadConfig();
    g_dev = reinterpret_cast<ID3D11Device *>(d->get_native());
    if (!InitNGX(g_dev)) {
        reshade::log::message(reshade::log::level::error, "AK DLAA: NGX init failed");
        Log("NGX init failed (is nvngx_dlss.dll next to the game exe?)");
        g_failed = true;
    }
}

static void on_destroy_device(device *) {
    if (g_dlss) NVSDK_NGX_D3D11_ReleaseFeature(g_dlss);
    if (g_ngx) NVSDK_NGX_D3D11_DestroyParameters(g_ngx);
    if (g_dev) NVSDK_NGX_D3D11_Shutdown1(g_dev);
    g_dlss = nullptr; g_ngx = nullptr;
    Rel(g_inTex); Rel(g_outTex); Rel(g_mvTex); Rel(g_mvUAV);
    Rel(g_mvCS); Rel(g_mvCB); Rel(g_depthSRV); Rel(g_depthTex);
    Rel(g_sharpTex); Rel(g_sharpUAV); Rel(g_outSRV); Rel(g_inSRV); Rel(g_postCS); Rel(g_postCB);
}

static void on_bind_rt(command_list *cmd, uint32_t, const resource_view *, resource_view dsv) {
    if (g_inside || !dsv.handle || !g_bbW) return;
    device *dev = cmd->get_device();
    resource r = dev->get_resource_from_view(dsv);
    ID3D11Texture2D *t = nullptr;
    if (FAILED(reinterpret_cast<ID3D11Resource *>(r.handle)->QueryInterface(__uuidof(ID3D11Texture2D), (void **)&t)) || !t)
        return;
    D3D11_TEXTURE2D_DESC d;
    t->GetDesc(&d);
    if (d.Format == DXGI_FORMAT_R32G8X24_TYPELESS && d.Width == g_bbW && d.Height == g_bbH && d.SampleDesc.Count == 1)
        SetDepth(t);
    t->Release();
}

static bool on_dispatch(command_list *cmd, uint32_t, uint32_t, uint32_t) {
    if (g_inside || !g_bbW) return false;
    ID3D11DeviceContext *ctx = reinterpret_cast<ID3D11DeviceContext *>(cmd->get_native());
    if (ctx->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) return false;
    ++g_dispatchCount;

    ID3D11ShaderResourceView *srv = nullptr;
    ctx->CSGetShaderResources(0, 1, &srv);
    if (!srv) return false;
    ID3D11Resource *res = nullptr;
    srv->GetResource(&res);
    srv->Release();
    if (!res) return false;
    ID3D11Texture2D *tex = nullptr;
    res->QueryInterface(__uuidof(ID3D11Texture2D), (void **)&tex);
    res->Release();
    if (!tex) return false;

    D3D11_TEXTURE2D_DESC d;
    tex->GetDesc(&d);
    bool match = d.Width == g_bbW && d.Height == g_bbH && (int)d.Format == cfg.colorFormat && d.SampleDesc.Count == 1;
    if ((int)g_frame == cfg.logFrame)
        Log("frame %u dispatch %u: SRV0 %ux%u format %u match=%d", g_frame, g_dispatchCount, d.Width, d.Height, (unsigned)d.Format, match ? 1 : 0);

    if (match) {
        uint32_t idx = g_matchCount++;
        if ((int)idx == cfg.triggerIndex && !g_done && g_haveView && g_enabled) {
            g_inside = true;
            RunDLAA(ctx, tex);
            g_inside = false;
        }
    }
    tex->Release();
    return false;
}

static void on_present(command_queue *, swapchain *sc, const rect *, const rect *, uint32_t, const rect *) {
    device *dev = sc->get_device();
    resource_desc rd = dev->get_resource_desc(sc->get_back_buffer(0));
    g_bbW = rd.texture.width;
    g_bbH = rd.texture.height;

    static bool prev9 = false, prev10 = false;
    bool d9 = (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
    bool d10 = (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
    if (d9 && !prev9) {
        g_enabled = !g_enabled;
        g_reset = true;
        Log("F9: DLAA %s", g_enabled ? "ON" : "OFF");
    }
    if (d10 && !prev10) {
        LoadConfig();
        g_w = 0;
        g_reset = true;
        g_failed = (g_ngx == nullptr);
        Log("F10: reloaded ini: TriggerIndex=%d HDR=%d Jitter=%d SignX=%.0f SignY=%.0f Swap=%d Sharp=%.2f Blend=%.2f Preset=%d",
            cfg.triggerIndex, cfg.hdrColor ? 1 : 0, cfg.applyJitter ? 1 : 0, cfg.jitterSignX, cfg.jitterSignY, cfg.swapMatrices ? 1 : 0, cfg.sharpness, cfg.blend, cfg.preset);
    }
    prev9 = d9;
    prev10 = d10;

    if ((int)g_frame == cfg.logFrame)
        Log("frame %u summary: backbuffer %ux%u, haveView=%d, depth=%s, dlaaDone=%d, near=%.2f, dispatches=%u, matches=%u, jitterApplied=%u times (jx=%.3f jy=%.3f)",
            g_frame, g_bbW, g_bbH, g_haveView ? 1 : 0, g_depthTex ? "found" : "MISSING", g_done ? 1 : 0, g_near, g_dispatchCount, g_matchCount, g_applyCount, g_jx, g_jy);

    ++g_frame;
    g_done = false;
    g_haveView = false;
    g_dispatchCount = 0;
    g_matchCount = 0;
    g_applyCount = 0;

    uint32_t i = (g_frame % (uint32_t)cfg.phases) + 1;
    g_jx = Halton(i, 2) - 0.5f;
    g_jy = Halton(i, 3) - 0.5f;
}

BOOL APIENTRY DllMain(HMODULE mod, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        if (!reshade::register_addon(mod)) return FALSE;
        reshade::register_event<reshade::addon_event::init_device>(on_init_device);
        reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
        reshade::register_event<reshade::addon_event::map_buffer_region>(on_map_buffer_region);
        reshade::register_event<reshade::addon_event::unmap_buffer_region>(on_unmap_buffer_region);
        reshade::register_event<reshade::addon_event::update_buffer_region>(on_update_buffer_region);
        reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_rt);
        reshade::register_event<reshade::addon_event::dispatch>(on_dispatch);
        reshade::register_event<reshade::addon_event::present>(on_present);
    } else if (reason == DLL_PROCESS_DETACH) {
        reshade::unregister_addon(mod);
    }
    return TRUE;
}
