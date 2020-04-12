# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:55:42 2019

@author: Dell
"""
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("C:/Users/Dell/Tensorflow/workspace/leaf_demo/trained_inference_graph/output_inference_graph_v1_lite.pb/saved_model",input_shapes={"image_tensor" : [1,300,300,3]})
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)