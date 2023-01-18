import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata

import tensorflow as tf

train_data = object_detector.DataLoader.from_pascal_voc(
    'data/traindata',
    'data/traindata',
    ['Ruger', 'Mom', 'Indy']
)

test_data = object_detector.DataLoader.from_pascal_voc(
    'data/testdata',
    'data/testdata',
    ['Ruger', 'Rody', 'Indy']
)

spec = model_spec.get('efficientdet_lite2')

model = object_objector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=24, validation_data=test_data)

model.evaluate(test_data)

model.export(export_dir='.', tflite_filename='dogs.tflite')