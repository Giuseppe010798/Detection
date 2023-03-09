from tensorflow.python.compiler.tensorrt import trt_convert as trt

SAVED_MODEL_DIR = "movenet_singlepose_lightning_4/"

# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=SAVED_MODEL_DIR)

# Convert the model into TRT compatible segments
trt_func = converter.convert()
# converter.summary()

OUTPUT_SAVED_MODEL_DIR = "tftrt_saved_model/"
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)
