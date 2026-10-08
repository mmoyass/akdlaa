// Arkham Knight DLAA prototype - ReShade addon (DX11) + NVIDIA NGX.
// Build as dlaa.addon64 (see CMakeLists.txt). UNTESTED SCAFFOLD.
//
// Sections marked [GAME-SPECIFIC] cannot be known without capturing the game
// in RenderDoc. Fill those in from your capture (see README.md).

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <reshade.hpp>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_params.h>
#include <nvsdk_ngx_helpers.h>
#include <cmath>
#include <cstdio>
#include <mutex>

extern "C" __declspec(dllexport) const char *NAME = "AK DLAA";
extern "C" __declspec(dllexport) const char *DESCRIPTION = "DLAA for Batman: Arkham Knight";

using namespace reshade::api;

// ---------------------------------------------------------------- config ----
// [GAME-SPECIFIC] Values to find in a RenderDoc capture, then put in dlaa.ini
struct Config {
    int      triggerDraw   = -1;  // draw-call index (per frame) where DLAA should run
    DXGI_FORMAT velocityFmt = DXGI_FORMAT_R16G16_FLOAT; // motion vector RT format
    bool     mvInPixels    = false; // true if vectors are already in pixels
    float    mvScaleX      = 1.0f;  // multiply to convert to pixels (usually -width or +width)
    float    mvScaleY      = 1.0f;
    bool     depthInverted = false;
    bool     hdrColor      = false; // true only if color RT is linear HDR (R11G11B10/FP16)
    float    preExposure   = 1.0f;
} cfg;

static void LoadConfig() {
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string p(path);
    p = p.substr(0, p.find_last_of("\\/") + 1) + "dlaa.ini";
    cfg.triggerDraw   = GetPrivateProfileIntA("dlaa", "TriggerDraw", -1, p.c_str());
    cfg.velocityFmt   = (DXGI_FORMAT)GetPrivateProfileIntA("dlaa", "VelocityFormat", (int)DXGI_FORMAT_R16G16_FLOAT, p.c_str());
    cfg.mvInPixels    = GetPrivateProfileIntA("dlaa", "MVInPixels", 0, p.c_str()) != 0;
    cfg.depthInverted = GetPrivateProfileIntA("dlaa", "DepthInverted", 0, p.c_str()) != 0;
    cfg.hdrColor      = GetPrivateProfileIntA("dlaa", "HDRColor", 0, p.c_str()) != 0;
    char buf[32];
    GetPrivateProfileStringA("dlaa", "MVScaleX", "1.0", buf, 32, p.c_str()); cfg.mvScaleX = (float)atof(buf);
    GetPrivateProfileStringA("dlaa", "MVScaleY", "1.0", buf, 32, p.c_str()); cfg.mvScaleY = (float)atof(buf);
}

// ----------------------------------------------------------------- state ----
static ID3D11Device        *g_dev   = nullptr;
static NVSDK_NGX_Parameter *g_ngx   = nullptr;
static NVSDK_NGX_Handle    *g_dlss  = nullptr;
static uint32_t g_w = 0, g_h = 0;
static ID3D11Texture2D *g_outTex = nullptr;   // DLAA output, copied back over the scene
static ID3D11Texture2D *g_inTex  = nullptr;   // copy of scene color (DLSS must not read+write one resource)
static ID3D11Resource  *g_velocity = nullptr;
static ID3D11Resource  *g_depth    = nullptr;
static uint32_t g_drawIndex = 0, g_frame = 0;
static bool     g_reset = true;
static float    g_jitterX = 0, g_jitterY = 0;

// Halton low-discrepancy sequence for sub-pixel jitter
static float Halton(uint32_t i, uint32_t b) {
    float f = 1, r = 0;
    while (i > 0) { f /= b; r += f * (i % b); i /= b; }
    return r;
}

// [GAME-SPECIFIC] Jitter must be applied to the SAME projection matrix the
// game uses for world rendering, otherwise DLSS sees wobble = flicker.
// Options: (a) patch the camera/projection constant write with a code hook
// (MinHook on the function that uploads the view-projection matrix),
// (b) modify the constant buffer contents in a map/update_buffer_region event.
// Jitter is in pixels; convert to clip space with  2*jx/width, -2*jy/height.
static void ApplyJitterToProjection(float /*jx_pixels*/, float /*jy_pixels*/) {
    // TODO: implement from your capture/reverse engineering.
}

static void NextJitter() {
    const uint32_t phases = 16;
    uint32_t i = (g_frame % phases) + 1;
    g_jitterX = Halton(i, 2) - 0.5f;
    g_jitterY = Halton(i, 3) - 0.5f;
    ApplyJitterToProjection(g_jitterX, g_jitterY);
}

// ------------------------------------------------------------------ NGX ----
static bool InitNGX(ID3D11Device *dev) {
    if (NVSDK_NGX_FAILED(NVSDK_NGX_D3D11_Init_with_ProjectID(
            "a0f57b54-1daf-4934-90ae-c4c0a0e1e7a1", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
            L".", dev)))
        return false;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_D3D11_GetCapabilityParameters(&g_ngx))) return false;
    int ok = 0;
    g_ngx->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &ok);
    return ok != 0;
}

static bool CreateDLSS(ID3D11DeviceContext *ctx, uint32_t w, uint32_t h) {
    if (g_dlss) { NVSDK_NGX_D3D11_ReleaseFeature(g_dlss); g_dlss = nullptr; }
    NVSDK_NGX_DLSS_Create_Params p = {};
    p.Feature.InWidth = p.Feature.InTargetWidth = w;     // render res == output res -> DLAA
    p.Feature.InHeight = p.Feature.InTargetHeight = h;
    p.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    p.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    if (cfg.hdrColor)      p.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    if (cfg.depthInverted) p.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    if (!cfg.mvInPixels)   p.InFeatureCreateFlags |= 0; // scale handled via MVScale at eval
    if (NVSDK_NGX_FAILED(NGX_D3D11_CREATE_DLSS_EXT(ctx, &g_dlss, g_ngx, &p))) return false;
    g_w = w; g_h = h; g_reset = true;
    return true;
}

static void MakeTex(ID3D11Texture2D *&tex, ID3D11Texture2D *like) {
    if (tex) { tex->Release(); tex = nullptr; }
    D3D11_TEXTURE2D_DESC d; like->GetDesc(&d);
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    d.MiscFlags = 0; d.SampleDesc = {1, 0}; d.Usage = D3D11_USAGE_DEFAULT; d.CPUAccessFlags = 0;
    g_dev->CreateTexture2D(&d, nullptr, &tex);
}

static void RunDLAA(ID3D11DeviceContext *ctx, ID3D11Texture2D *sceneColor) {
    if (!g_dlss || !g_velocity || !g_depth) return;
    ctx->CopyResource(g_inTex, sceneColor);

    NVSDK_NGX_D3D11_DLSS_Eval_Params e = {};
    e.Feature.pInColor  = g_inTex;
    e.Feature.pInOutput = g_outTex;
    e.pInDepth          = g_depth;
    e.pInMotionVectors  = g_velocity;
    e.InJitterOffsetX   = g_jitterX;
    e.InJitterOffsetY   = g_jitterY;
    e.InRenderSubrectDimensions = {g_w, g_h};
    e.InReset           = g_reset ? 1 : 0;
    e.InMVScaleX        = cfg.mvInPixels ? 1.0f : cfg.mvScaleX * (float)g_w;
    e.InMVScaleY        = cfg.mvInPixels ? 1.0f : cfg.mvScaleY * (float)g_h;
    e.InPreExposure     = cfg.preExposure;
    e.Feature.InSharpness = 0.0f;

    if (NVSDK_NGX_SUCCEED(NGX_D3D11_EVALUATE_DLSS_EXT(ctx, g_dlss, g_ngx, &e))) {
        ctx->CopyResource(sceneColor, g_outTex);
        g_reset = false;
    }
}

// ---------------------------------------------------------------- events ----
static void on_init_device(device *d) {
    LoadConfig();
    g_dev = reinterpret_cast<ID3D11Device *>(d->get_native());
    if (!InitNGX(g_dev)) reshade::log::message(reshade::log::level::error, "AK DLAA: NGX init failed");
}

static void on_destroy_device(device *) {
    if (g_dlss) NVSDK_NGX_D3D11_ReleaseFeature(g_dlss);
    if (g_ngx)  NVSDK_NGX_D3D11_DestroyParameters(g_ngx);
    NVSDK_NGX_D3D11_Shutdown1(g_dev);
    g_dlss = nullptr; g_ngx = nullptr;
    if (g_outTex) g_outTex->Release();
    if (g_inTex)  g_inTex->Release();
}

// Track depth + velocity as the game binds render targets.
static void on_bind_rt(command_list *cmd, uint32_t count, const resource_view *rtvs, resource_view dsv) {
    device *dev = cmd->get_device();
    if (dsv.handle) {
        resource r = dev->get_resource_from_view(dsv);
        // [GAME-SPECIFIC] pick the main scene depth (largest D24S8/D32 matching backbuffer size)
        g_depth = reinterpret_cast<ID3D11Resource *>(r.handle);
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (!rtvs[i].handle) continue;
        resource r = dev->get_resource_from_view(rtvs[i]);
        resource_desc d = dev->get_resource_desc(r);
        // [GAME-SPECIFIC] identify the velocity target (format+size is a good start;
        // confirm in RenderDoc which draw writes it, add more checks if needed)
        if (d.type == resource_type::texture_2d && (DXGI_FORMAT)d.texture.format == cfg.velocityFmt)
            g_velocity = reinterpret_cast<ID3D11Resource *>(r.handle);
    }
}

static bool on_draw(command_list *cmd, uint32_t, uint32_t, uint32_t, uint32_t) {
    if (cfg.triggerDraw >= 0 && (int)g_drawIndex == cfg.triggerDraw) {
        // [GAME-SPECIFIC] this should be the pass right where the game would do its own
        // AA/temporal resolve: scene color complete, before tonemap/UI/motion blur.
        // Here we assume the currently bound RT0 is the scene color.
        // (Retrieve it via a tracked RTV from on_bind_rt in a real implementation.)
    }
    ++g_drawIndex;
    return false; // don't block the draw
}

static void on_present(command_queue *, swapchain *) {
    g_drawIndex = 0;
    ++g_frame;
    NextJitter();
}

BOOL APIENTRY DllMain(HMODULE mod, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        if (!reshade::register_addon(mod)) return FALSE;
        reshade::register_event<reshade::addon_event::init_device>(on_init_device);
        reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
        reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_rt);
        reshade::register_event<reshade::addon_event::draw>(on_draw);
        reshade::register_event<reshade::addon_event::present>(on_present);
    } else if (reason == DLL_PROCESS_DETACH) {
        reshade::unregister_addon(mod);
    }
    return TRUE;
}
