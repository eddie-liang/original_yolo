import coremltools

coreml_model=coremltools.converters.keras.convert(
        'yad2k/model_data/tiny-yolo-voc.h5',
        input_names='image',
        image_input_names='image',
        output_names='grid',
        image_scale=1/255.)

coreml_model.input_description['image']='Input image'
coreml_model.output_description['grid']='The 13*13 grid'

coreml_model.save('../TinyYOLO-CoreML/TinyYOLO-CoreML/TinyYOLO.mlmodel')