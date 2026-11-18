# The audio front end of TensorFlow Lite Micro (window, FFT, mel filterbank,
# noise reduction, gain control, log scale), for the ESP32-S3 build.
file(GLOB sources
    "tensorflow/lite/experimental/microfrontend/lib/*.c"
    "tensorflow/lite/experimental/microfrontend/lib/*.cpp"
    "third_party/kissfft/kiss_fft.c"
    "third_party/kissfft/tools/kiss_fftr.c")

idf_component_register(SRCS ${sources} INCLUDE_DIRS ".")

# Third-party code: do not stop the build on its warnings.
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error -Wno-unused-parameter)
