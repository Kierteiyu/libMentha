# libMentha

**libMentha** is a continuation of the [**tiny-dnn**](https://github.com/tiny-dnn/tiny-dnn) machine learning library.

## Planned Features

1. **GPU Support** via Vulkan Compute via [Kompute](https://kompute.cc)
2. **More Layers** (Open an issue to request a layer. It might take a while as I have other priorities right now)
3. **ONNX Saving and Loading**
4. **Improved Serialization System**

---

## Current Version: `v0.2.0`

This version is essentially a cleaner version of the tiny-dnn source tree, with C++ 20 support.

---

## Upcoming Version: `v0.4.0`

Planned improvements:

- Unknown

---

## Usage and Installation

Place `lib_mentha_include` in an accessible location on your drive. Configure your build system to include `lib_mentha_include`. If you want serialization, do the same with [cereal](https://github.com/USCiLab/cereal) according to their instructions. If you don't want serialization, define `CNN_NO_SERIALIZATION`.

---

## Notes

- tiny-dnn examples should work by find and replacing tiny-dnn with lib_mentha.
