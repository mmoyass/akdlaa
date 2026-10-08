# Arkham Knight DLAA (prototype scaffold)

Status: **untested scaffold.** It compiles in principle against the ReShade addon API
and NVIDIA NGX SDK, but the game-specific parts are intentionally left as TODOs because
they can only be found by inspecting the real game.

## Why this design
- A ReShade *addon* gives clean DX11 access (resources, render-target binds, draw calls)
  without writing a raw D3D11 hook.
- DLAA = DLSS with render resolution == output resolution.
- DLSS flickers when any of these are wrong: **jitter**, **motion vectors (sign/scale)**,
  **depth (inverted?)**, **color space (HDR vs LDR)**, or **where in the frame it runs**.
  Each is a setting in `dlaa.ini` or a marked TODO in the code.

## Build
1. Get the ReShade source headers (`reshade.hpp`, from github.com/crosire/reshade, `include/`).
2. Get the NVIDIA DLSS SDK (github.com/NVIDIA/DLSS) - need `sdk/include`, `nvsdk_ngx_d.lib`,
   and `nvngx_dlss.dll` (put the DLL next to the game exe).
3. `cmake -B build -DRESHADE_INC=... -DNGX_ROOT=...` then `cmake --build build --config Release`
4. Copy `dlaa.addon64` next to the game exe (use ReShade **with addon support**, DX11).

## The game-specific work (do this in RenderDoc)
Capture a frame of Arkham Knight (DX11, windowed, ReShade off for the capture), then find:

| Need | How to find it | Where it goes |
|---|---|---|
| Scene color before AA/tonemap/UI | Last pass before the game's AA/post | `TriggerDraw` (draw index) |
| Depth | Main D24S8/D32 matching window size; check if near=1 (inverted) | `DepthInverted` |
| Motion vectors | The velocity RT used by motion blur (often R16G16 or R8G8) | `VelocityFormat`, `MVScaleX/Y` |
| Projection matrix upload | Constant buffer holding view-projection | `ApplyJitterToProjection()` |

## dlaa.ini example
```
[dlaa]
TriggerDraw=1234
VelocityFormat=34
MVInPixels=0
MVScaleX=-1.0
MVScaleY=-1.0
DepthInverted=0
HDRColor=0
```

## Debugging flicker
- Shimmer on edges while moving: motion vector sign or scale wrong -> flip `MVScaleX/Y`.
- Whole-image wobble/shake: jitter not applied to projection (or applied to the wrong matrix).
- Ghost trails: depth inverted setting wrong, or velocity missing for moving objects/skinned meshes.
- Flashing brightness: running after tonemap with `HDRColor=1`, or auto-exposure vs LDR mismatch.
- Blur on UI: trigger point is after UI draw; move `TriggerDraw` earlier.
