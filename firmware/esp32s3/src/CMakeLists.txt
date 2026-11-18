# Sources of the ESP32-S3 firmware (ESP-IDF): the board's own files in this
# folder, the code shared with the Nano 33 BLE Sense in ../../common, and the
# model of the PlatformIO environment.
#
# PlatformIO builds each environment in .pio/build/<name>, so the name of the
# build folder is the name of the environment, and the model is
# ../../models/<name>/.

set(firmware_dir "${CMAKE_CURRENT_LIST_DIR}/../..")
set(common_dir "${firmware_dir}/common")

set(model_dir "")
if(NOT CMAKE_BUILD_EARLY_EXPANSION)
    get_filename_component(environment "${CMAKE_BINARY_DIR}" NAME)
    set(model_dir "${firmware_dir}/models/${environment}")
    if(NOT EXISTS "${model_dir}/model_data.cpp")
        message(FATAL_ERROR "No model in ${model_dir} for the environment "
                            "${environment}; see firmware/esp32s3/README.md")
    endif()
endif()

idf_component_register(
    SRCS
        "audio_provider.cpp"
        "board.cpp"
        "lamp_output.cpp"
        "${common_dir}/main.cpp"
        "${common_dir}/lamp_output.cpp"
        "${common_dir}/lamp_recognizer.cpp"
        "${common_dir}/feature_provider.cpp"
        "${common_dir}/micro_features_generator.cpp"
        "${common_dir}/microphone.cpp"
        "${model_dir}/model_data.cpp"
    INCLUDE_DIRS "." "${common_dir}" "${model_dir}"
    PRIV_REQUIRES esp_driver_i2s esp_timer microfrontend esp-tflite-micro led_strip)
