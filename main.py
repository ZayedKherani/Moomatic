from flask import Flask, render_template, Markup
from flask import __version__ as flaskVersion
import tensorflow_transform.beam as tft_beam
from os.path import exists as pathExists
from datetime import datetime, timedelta
from os import mkdir, getcwd, remove
from os.path import join as pathJoin
from random import choice, randint
import tensorflow_transform as tft
from json import dump as jsonDump
from json import load as jsonLoad
from tfx_bsl.public import tfxio
import apache_beam as beam
from shutil import rmtree
import tensorflow as tf
import pprint
import string
import math
import sys

# Confirm that we're using Python 3
assert sys.version_info.major == 3, 'Oops, not running Python 3.'

print('TF: {}'.format(tf.__version__))
print('Flask: {}'.format(flaskVersion))
print('Beam: {}'.format(beam.__version__))
print('Transform: {}'.format(tft.__version__))

mainTempDir = pathJoin(getcwd(), 'temps')
rmtree(mainTempDir) if pathExists(mainTempDir) else ''
mkdir(mainTempDir)

if pathExists('./adult2.data'):
    remove('./adult2.data')

rootDay = {}

app = Flask(__name__)

train = './adult2.data'
test = './adult2.test'

CATEGORICAL_FEATURE_KEYS = [
    'age',
    'tOF',
    'tsYDM0',
    'tsHM0',
    'tsS0',
    'tsYDM1',
    'tsHM1',
    'tsS2',
]

NUMERIC_FEATURE_KEYS = [
    'id',
    'tsS1',
    'tsYDM2',
    'tsHM2',
]

OPTIONAL_NUMERIC_FEATURE_KEYS = [
    'aE012',
]

LABEL_KEY = 'label'

ORDERED_CSV_COLUMNS = [
    'id', 'age', 'wgt', 'tOF', 'aE012',
    'tsYDM0', 'tsHM0', 'tsS0', 'tsYDM1', 'tsHM1',
    'tsS1', 'tsYDM2', 'tsHM2', 'tsS2', 'label',
]


RAW_DATA_FEATURE_SPEC = dict(
    [(name, tf.io.FixedLenFeature([], tf.string))
     for name in CATEGORICAL_FEATURE_KEYS] +
    [(name, tf.io.FixedLenFeature([], tf.float32))
     for name in NUMERIC_FEATURE_KEYS] +
    [(name, tf.io.VarLenFeature(tf.float32))
     for name in OPTIONAL_NUMERIC_FEATURE_KEYS] +
    [(LABEL_KEY, tf.io.FixedLenFeature([], tf.string))]
)

SCHEMA = tft.tf_metadata.dataset_metadata.DatasetMetadata(
    tft.tf_metadata.schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC)).schema

NUM_OOV_BUCKETS = 1
TRAIN_NUM_EPOCHS = 16
NUM_TRAIN_INSTANCES = 24
TRAIN_BATCH_SIZE = 128
NUM_TEST_INSTANCES = 7

# Names of temp files
TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
EXPORTED_MODEL_DIR = 'exported_model_dir'


def genDataOneCow(idIteration: int):
    # generate id
    id: int = idIteration + 1

    # generate age
    age: int = randint(0, 22)

    # generate weight
    if 0 <= age <= 7:
        weight = randint(200, 900)
    elif 7 < age <= 22:
        weight = randint(900, 1400)

    # generate type of feed
    typeOfFeed = randint(1, 4)

    # ammount eaten
    ammountEatenBreakfast = randint(20, 30)
    ammountEatenLunch = randint(20, 30)
    ammountEatenDinner = randint(20, 30)

    # generate time stamps
    currentDateTime = datetime.now()

    timestampBreakfast = {
        "year": currentDateTime.year,
        "month": currentDateTime.month,
        "day": currentDateTime.day,
        "hour": 6,
        "minute": 0,
        "second": 0
    }

    timestampLunch = {
        "year": currentDateTime.year,
        "month": currentDateTime.month,
        "day": currentDateTime.day,
        "hour": 12,
        "minute": 0,
        "second": 0
    }

    timestampDinner = {
        "year": currentDateTime.year,
        "month": currentDateTime.month,
        "day": currentDateTime.day,
        "hour": 18,
        "minute": 0,
        "second": 0
    }

    label = 1 if age > 7 else 0

    currentRootData = {
        "ID": id,
        "Age": age,
        "Weight": weight,
        "Type of Feed": typeOfFeed,
        "Ammount Eaten": {
            0: ammountEatenBreakfast,
            1: ammountEatenLunch,
            2: ammountEatenDinner
        },
        "Timestamp": {
            0: timestampBreakfast,
            1: timestampLunch,
            2: timestampDinner
        },
        "Label": label
    }

    return currentRootData


def genDataWeekCow():
    for i in range(NUM_TRAIN_INSTANCES):
        rootDay.update({i: genDataOneCow(i)})


def numberToLetter(number):
    lTNdic = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }

    number = str(number)

    numberStr = ""

    for i in number:
        numberStr += lTNdic[int(i)] + "-"

    numberStr = numberStr[:-1]

    return numberStr


def writeToFile():
    file = open('adult2.data', 'a')

    for i in range(len(rootDay)):
        writeString = "{id}, {age}, {wgt}, {tOF}, {aE012}, {tsYMD0}, {tsHM0}, {tsS0}, {tsYMD1}, {tsHM1}, {tsS1}, {tsYMD2}, {tsHM2}, {tsS2}, {label_key}.\n".format(
            id=rootDay[i]["ID"],
            age=numberToLetter(rootDay[i]["Age"]),
            wgt=rootDay[i]["Weight"],
            tOF=numberToLetter(rootDay[i]["Type of Feed"]),
            aE012=(((rootDay[i]["Ammount Eaten"][0] * 100) + rootDay[i]
                    ["Ammount Eaten"][1]) * 100) + rootDay[i]["Ammount Eaten"][2],
            tsYMD0=numberToLetter((((rootDay[i]["Timestamp"][0]["year"] * 100) + rootDay[i]
                                    ["Timestamp"][0]["month"]) * 100) + rootDay[i]["Timestamp"][0]["day"]),
            tsHM0=numberToLetter(
                (rootDay[i]["Timestamp"][0]["hour"] * 100) + rootDay[i]["Timestamp"][0]["minute"]),
            tsS0=numberToLetter(rootDay[i]["Timestamp"][0]["second"]),
            tsYMD1=numberToLetter((((rootDay[i]["Timestamp"][1]["year"] * 100) + rootDay[i]
                                    ["Timestamp"][1]["month"]) * 100) + rootDay[i]["Timestamp"][2]["day"]),
            tsHM1=numberToLetter(
                (rootDay[i]["Timestamp"][1]["hour"] * 100) + rootDay[i]["Timestamp"][1]["minute"]),
            tsS1=rootDay[i]["Timestamp"][1]["second"],
            tsYMD2=(((rootDay[i]["Timestamp"][2]["year"] * 100) + rootDay[i]
                     ["Timestamp"][2]["month"]) * 100) + rootDay[i]["Timestamp"][1]["day"],
            tsHM2=(rootDay[i]["Timestamp"][2]["hour"] * 100) +
            rootDay[i]["Timestamp"][2]["minute"],
            tsS2=numberToLetter(rootDay[i]["Timestamp"][2]["second"]),
            label_key=rootDay[i]["Label"]
        )

        file.write(writeString)

    file.close()


def makeTempDirectory():
    """Make a local temporary directory for the program to access."""

    # Generate the directory path
    directoryName = "".join(choice(string.ascii_letters + string.digits)
                            for x in range(randint(8, 16)))
    directoryPath = pathJoin(mainTempDir, directoryName)

    # Make the directory
    mkdir(directoryPath)

    # return the directory path
    return directoryPath


def preprocessor(inputs):
    """Preprocess input columns into transformed columns."""
    # Since we are modifying some features and leaving others unchanged, we
    # start by setting `outputs` to a copy of `inputs.
    outputs = inputs.copy()

    # Scale numeric columns to have range [0, 1].
    for key in NUMERIC_FEATURE_KEYS:
        outputs[key] = tft.scale_to_0_1(inputs[key])

    for key in OPTIONAL_NUMERIC_FEATURE_KEYS:
        # This is a SparseTensor because it is optional. Here we fill in a default
        # value when it is missing.
        sparse = tf.sparse.SparseTensor(inputs[key].indices, inputs[key].values,
                                        [inputs[key].dense_shape[0], 1])
        dense = tf.sparse.to_dense(sp_input=sparse, default_value=0.)
        # Reshaping from a batch of vectors of size 1 to a batch to scalars.
        dense = tf.squeeze(dense, axis=1)
        outputs[key] = tft.scale_to_0_1(dense)

    # For all categorical columns except the label column, we generate a
    # vocabulary but do not modify the feature.  This vocabulary is instead
    # used in the trainer, by means of a feature column, to convert the feature
    # from a string to an integer id.
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[key] = tft.compute_and_apply_vocabulary(
            tf.strings.strip(inputs[key]),
            num_oov_buckets=NUM_OOV_BUCKETS,
            vocab_filename=key)

    # For the label column we provide the mapping from string to index.
    table_keys = ['0', '1']
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=table_keys,
        values=tf.cast(tf.range(len(table_keys)), tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64)
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    # Remove trailing periods for test data when the data is read with tf.data.
    label_str = tf.strings.regex_replace(inputs[LABEL_KEY], r'\.', '')
    label_str = tf.strings.strip(label_str)
    data_labels = table.lookup(label_str)
    transformed_label = tf.one_hot(
        indices=data_labels, depth=len(table_keys), on_value=1.0, off_value=0.0)
    outputs[LABEL_KEY] = tf.reshape(transformed_label, [-1, len(table_keys)])

    return outputs


def transformer(train_data_file, test_data_file, working_dir):
    """Transform the data and write out as a TFRecord of Example protos.

    Read in the data using the CSV reader, and transform it using a
    preprocessing pipeline that scales numeric data and converts categorical data
    from strings to int64 values indices, by creating a vocabulary for each
    category.

    Args:
      train_data_file: File containing training data
      test_data_file: File containing test data
      working_dir: Directory to write transformed data and metadata to
    """

    # The "with" block will create a pipeline, and run that pipeline at the exit
    # of the block.
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir=makeTempDirectory()):
            # Create a TFXIO to read the census data with the schema. To do this we
            # need to list all columns in order since the schema doesn't specify the
            # order of columns in the csv.
            # We first read CSV files and use BeamRecordCsvTFXIO whose .BeamSource()
            # accepts a PCollection[bytes] because we need to patch the records first
            # (see "FixCommasTrainData" below). Otherwise, tfxio.CsvTFXIO can be used
            # to both read the CSV files and parse them to TFT inputs:
            # csv_tfxio = tfxio.CsvTFXIO(...)
            # raw_data = (pipeline | 'ToRecordBatches' >> csv_tfxio.BeamSource())
            csv_tfxio = tfxio.BeamRecordCsvTFXIO(
                physical_format='text',
                column_names=ORDERED_CSV_COLUMNS,
                schema=SCHEMA)

            # Read in raw data and convert using CSV TFXIO.  Note that we apply
            # some Beam transformations here, which will not be encoded in the TF
            # graph since we don't do the from within tf.Transform's methods
            # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
            # to get data into a format that the CSV TFXIO can read, in particular
            # removing spaces after commas.
            raw_data = (
                pipeline
                | 'ReadTrainData' >> beam.io.ReadFromText(
                    train_data_file, coder=beam.coders.BytesCoder())
                | 'FixCommasTrainData' >> beam.Map(
                    lambda line: line.replace(b', ', b','))
                | 'DecodeTrainData' >> csv_tfxio.BeamSource())

            # Combine data and schema into a dataset tuple.  Note that we already used
            # the schema to read the CSV data, but we also need it to interpret
            # raw_data.
            raw_dataset = (raw_data, csv_tfxio.TensorAdapterConfig())
            transformed_dataset, transform_fn = (
                raw_dataset | tft_beam.AnalyzeAndTransformDataset(preprocessor))
            transformed_data, transformed_metadata = transformed_dataset
            transformed_data_coder = tft.coders.ExampleProtoCoder(
                transformed_metadata.schema)

            _ = (
                transformed_data
                | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
                | 'WriteTrainData' >> beam.io.WriteToTFRecord(
                    pathJoin(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

            # Now apply transform function to test data.  In this case we remove the
            # trailing period at the end of each line, and also ignore the header line
            # that is present in the test data file.
            raw_test_data = (
                pipeline
                | 'ReadTestData' >> beam.io.ReadFromText(
                    test_data_file, skip_header_lines=1,
                    coder=beam.coders.BytesCoder())
                | 'FixCommasTestData' >> beam.Map(
                    lambda line: line.replace(b', ', b','))
                | 'RemoveTrailingPeriodsTestData' >> beam.Map(lambda line: line[:-1])
                | 'DecodeTestData' >> csv_tfxio.BeamSource())

            raw_test_dataset = (raw_test_data, csv_tfxio.TensorAdapterConfig())

            transformed_test_dataset = (
                (raw_test_dataset, transform_fn) | tft_beam.TransformDataset())
            # Don't need transformed data schema, it's the same as before.
            transformed_test_data, _ = transformed_test_dataset

            _ = (
                transformed_test_data
                | 'EncodeTestData' >> beam.Map(transformed_data_coder.encode)
                | 'WriteTestData' >> beam.io.WriteToTFRecord(
                    pathJoin(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)))

            # Will write a SavedModel and metadata to working_dir, which can then
            # be read by the tft.TFTransformOutput class.
            _ = (
                transform_fn
                | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))


def preprocessTrainerInput(tf_transform_output, transformed_examples,
                           batch_size):
    """An input function reading from transformed data, converting to model input.

    Args:
      tf_transform_output: Wrapper around output of tf.Transform.
      transformed_examples: Base filename of examples.
      batch_size: Batch size.

    Returns:
      The input data for training or eval, in the form of k.
    """
    def input_fn():
        return tf.data.experimental.make_batched_features_dataset(
            file_pattern=transformed_examples,
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            label_key=LABEL_KEY,
            shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)

    return input_fn


def preprocessServerInput(tf_transform_output, raw_examples, batch_size):
    """An input function reading from raw data, converting to model input.

    Args:
      tf_transform_output: Wrapper around output of tf.Transform.
      raw_examples: Base filename of examples.
      batch_size: Batch size.

    Returns:
      The input data for training or eval, in the form of k.
    """

    def get_ordered_raw_data_dtypes():
        result = []
        for col in ORDERED_CSV_COLUMNS:
            if col not in RAW_DATA_FEATURE_SPEC:
                result.append(0.0)
                continue
            spec = RAW_DATA_FEATURE_SPEC[col]
            if isinstance(spec, tf.io.FixedLenFeature):
                result.append(spec.dtype)
            else:
                result.append(0.0)
        return result

    def input_fn():
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=raw_examples,
            batch_size=batch_size,
            column_names=ORDERED_CSV_COLUMNS,
            column_defaults=get_ordered_raw_data_dtypes(),
            prefetch_buffer_size=0,
            ignore_errors=True)

        tft_layer = tf_transform_output.transform_features_layer()

        def transform_dataset(data):
            raw_features = {}
            for key, val in data.items():
                if key not in RAW_DATA_FEATURE_SPEC:
                    continue
                if isinstance(RAW_DATA_FEATURE_SPEC[key], tf.io.VarLenFeature):
                    raw_features[key] = tf.RaggedTensor.from_tensor(
                        tf.expand_dims(val, -1)).to_sparse()
                    continue
                raw_features[key] = val
            transformed_features = tft_layer(raw_features)
            data_labels = transformed_features.pop(LABEL_KEY)
            return (transformed_features, data_labels)

        return dataset.map(
            transform_dataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
                tf.data.experimental.AUTOTUNE)

    return input_fn


def preprocessServerModelExporter(tf_transform_output, model, output_dir):
    """Exports a keras model for serving.

    Args:
      tf_transform_output: Wrapper around output of tf.Transform.
      model: A keras model to export for serving.
      output_dir: A directory where the model will be exported to.
    """
    # The layer has to be saved to the model for keras tracking purpases.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Serving tf.function model wrapper."""
        feature_spec = RAW_DATA_FEATURE_SPEC.copy()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)
        classes_names = tf.constant([['0', '1']])
        classes = tf.tile(classes_names, [tf.shape(outputs)[0], 1])
        return {'classes': classes, 'scores': outputs}

    concrete_serving_fn = serve_tf_examples_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='inputs'))
    signatures = {'serving_default': concrete_serving_fn}

    # This is required in order to make this model servable with model_server.
    versioned_output_dir = pathJoin(output_dir, '1')
    model.save(versioned_output_dir, save_format='tf', signatures=signatures)


def preprocessTrainer(working_dir,
                      num_train_instances=NUM_TRAIN_INSTANCES,
                      num_test_instances=NUM_TEST_INSTANCES):
    """Train the model on training data and evaluate on test data.

    Args:
      working_dir: The location of the Transform output.
      num_train_instances: Number of instances in train set
      num_test_instances: Number of instances in test set

    Returns:
      The results from the estimator's 'evaluate' method
    """
    train_data_path_pattern = pathJoin(working_dir,
                                       TRANSFORMED_TRAIN_DATA_FILEBASE + '*')
    eval_data_path_pattern = pathJoin(working_dir,
                                      TRANSFORMED_TEST_DATA_FILEBASE + '*')
    tf_transform_output = tft.TFTransformOutput(working_dir)

    train_input_fn = preprocessTrainerInput(
        tf_transform_output, train_data_path_pattern, batch_size=TRAIN_BATCH_SIZE)
    train_dataset = train_input_fn()

    # Evaluate model on test dataset.
    eval_input_fn = preprocessTrainerInput(
        tf_transform_output, eval_data_path_pattern, batch_size=TRAIN_BATCH_SIZE)
    validation_dataset = eval_input_fn()

    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(LABEL_KEY)

    inputs = {}
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.VarLenFeature):
            inputs[key] = tf.keras.layers.Input(
                shape=[None], name=key, dtype=spec.dtype, sparse=True)
        elif isinstance(spec, tf.io.FixedLenFeature):
            inputs[key] = tf.keras.layers.Input(
                shape=spec.shape, name=key, dtype=spec.dtype)
        else:
            raise ValueError('Spec type is not supported: ', key, spec)

    encoded_inputs = {}
    for key in inputs:
        feature = tf.expand_dims(inputs[key], -1)
        if key in CATEGORICAL_FEATURE_KEYS:
            num_buckets = tf_transform_output.num_buckets_for_transformed_feature(
                key)
            encoding_layer = (
                tf.keras.layers.experimental.preprocessing.CategoryEncoding(
                    max_tokens=num_buckets, output_mode='binary', sparse=False))
            encoded_inputs[key] = encoding_layer(feature)
        else:
            encoded_inputs[key] = feature

    stacked_inputs = tf.concat(tf.nest.flatten(encoded_inputs), axis=1)
    output = tf.keras.layers.Dense(100, activation='relu')(stacked_inputs)
    output = tf.keras.layers.Dense(70, activation='relu')(output)
    output = tf.keras.layers.Dense(50, activation='relu')(output)
    output = tf.keras.layers.Dense(20, activation='relu')(output)
    output = tf.keras.layers.Dense(2, activation='sigmoid')(output)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    pprint.pprint(model.summary())

    model.fit(train_dataset, validation_data=validation_dataset,
              epochs=TRAIN_NUM_EPOCHS,
              steps_per_epoch=math.ceil(
                  num_train_instances / TRAIN_BATCH_SIZE),
              validation_steps=math.ceil(num_test_instances / TRAIN_BATCH_SIZE))

    # Export the model.
    exported_model_dir = pathJoin(working_dir, EXPORTED_MODEL_DIR)
    preprocessServerModelExporter(
        tf_transform_output, model, exported_model_dir)

    metrics_values = model.evaluate(
        validation_dataset, steps=num_test_instances)
    metrics_labels = model.metrics_names
    return {l: v for l, v in zip(metrics_labels, metrics_values)}


def trainerTrainerInput(tf_transform_output, transformed_examples,
                        batch_size):
    """Creates an input function reading from transformed data.

    Args:
      tf_transform_output: Wrapper around output of tf.Transform.
      transformed_examples: Base filename of examples.
      batch_size: Batch size.

    Returns:
      The input function for training or eval.
    """
    def input_fn():
        """Input function for training and eval."""
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=transformed_examples,
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            shuffle=True)

        transformed_features = tf.compat.v1.data.make_one_shot_iterator(
            dataset).get_next()

        # Extract features and label from the transformed tensors.
        transformed_labels = tf.where(
            tf.equal(transformed_features.pop(LABEL_KEY), 1))

        return transformed_features, transformed_labels[:, 1]

    return input_fn


def trainerServerInput(tf_transform_output):
    """Creates an input function reading from raw data.

    Args:
      tf_transform_output: Wrapper around output of tf.Transform.

    Returns:
      The serving input function.
    """
    raw_feature_spec = RAW_DATA_FEATURE_SPEC.copy()
    # Remove label since it is not available during serving.
    raw_feature_spec.pop(LABEL_KEY)

    def serving_input_fn():
        """Input function for serving."""
        # Get raw features by generating the basic serving input_fn and calling it.
        # Here we generate an input_fn that expects a parsed Example proto to be fed
        # to the model at serving time.  See also
        # tf.estimator.export.build_raw_serving_input_receiver_fn.
        raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            raw_feature_spec, default_batch_size=None)
        serving_input_receiver = raw_input_fn()

        # Apply the transform function that was used to generate the materialized
        # data.
        raw_features = serving_input_receiver.features
        transformed_features = tf_transform_output.transform_raw_features(
            raw_features)

        return tf.estimator.export.ServingInputReceiver(
            transformed_features, serving_input_receiver.receiver_tensors)

    return serving_input_fn


def trainerGetFeatureColumns(tf_transform_output):
    """Returns the FeatureColumns for the model.

    Args:
      tf_transform_output: A `TFTransformOutput` object.

    Returns:
      A list of FeatureColumns.
    """
    # Wrap scalars as real valued columns.
    real_valued_columns = [tf.feature_column.numeric_column(key, shape=())
                           for key in NUMERIC_FEATURE_KEYS]

    # Wrap categorical columns.
    one_hot_columns = [
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(
                key=key,
                num_buckets=(NUM_OOV_BUCKETS +
                             tf_transform_output.vocabulary_size_by_name(
                                 vocab_filename=key))))
        for key in CATEGORICAL_FEATURE_KEYS]

    return real_valued_columns + one_hot_columns


def trainerTrainer(working_dir, num_train_instances=NUM_TRAIN_INSTANCES,
                   num_test_instances=NUM_TEST_INSTANCES):
    """Train the model on training data and evaluate on test data.

    Args:
      working_dir: Directory to read transformed data and metadata from and to
          write exported model to.
      num_train_instances: Number of instances in train set
      num_test_instances: Number of instances in test set

    Returns:
      The results from the estimator's 'evaluate' method
    """
    tf_transform_output = tft.TFTransformOutput(working_dir)

    run_config = tf.estimator.RunConfig()

    estimator = tf.estimator.LinearClassifier(
        feature_columns=trainerGetFeatureColumns(tf_transform_output),
        config=run_config,
        loss_reduction=tf.losses.Reduction.SUM)

    # Fit the model using the default optimizer.
    train_input_fn = trainerTrainerInput(
        tf_transform_output,
        pathJoin(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE + '*'),
        batch_size=TRAIN_BATCH_SIZE)
    estimator.train(
        input_fn=train_input_fn,
        max_steps=TRAIN_NUM_EPOCHS * num_train_instances / TRAIN_BATCH_SIZE)

    # Evaluate model on test dataset.
    eval_input_fn = trainerTrainerInput(
        tf_transform_output,
        pathJoin(working_dir, TRANSFORMED_TEST_DATA_FILEBASE + '*'),
        batch_size=1)

    # Export the model.
    serving_input_fn = trainerServerInput(tf_transform_output)
    exported_model_dir = pathJoin(working_dir, EXPORTED_MODEL_DIR)
    estimator.export_saved_model(exported_model_dir, serving_input_fn)

    return estimator.evaluate(input_fn=eval_input_fn, steps=num_test_instances)


def generator():
    genDataWeekCow()

    with open("cowData.json", "w") as jsonFile:
        jsonDump(rootDay, jsonFile)

    writeToFile()


def generateTimeStamp(i):
    currentCowData = cowData[i]

    currentDateTime = datetime.now()

    if currentCowData["Timestamp"]["2"]["minute"] == 6:
        hour = 12
    elif currentCowData["Timestamp"]["2"]["minute"] == 12:
        hour = 18
    else:
        hour = 6

    if (currentCowData["Timestamp"]["2"]["year"] == currentCowData["Timestamp"]["1"]["year"] == currentCowData["Timestamp"]["0"]["year"]) and (currentCowData["Timestamp"]["2"]["month"] == currentCowData["Timestamp"]["1"]["month"] == currentCowData["Timestamp"]["0"]["month"]) and (currentCowData["Timestamp"]["2"]["day"] == currentCowData["Timestamp"]["1"]["day"] == currentCowData["Timestamp"]["0"]["day"]):
        timestampDateTime = datetime(year=currentCowData["Timestamp"]["2"]["year"], month=currentCowData["Timestamp"]
                                     ["2"]["month"], day=currentCowData["Timestamp"]["2"]["day"]) + timedelta(days=1)

        timestamp = {
            "year": timestampDateTime.year,
            "month": timestampDateTime.month,
            "day": timestampDateTime.day,
            "hour": hour,
            "minute": 0,
            "second": 0
        }
    else:
        currentDateTime = datetime.now()

        timestamp = {
            "year": currentDateTime.year,
            "month": currentDateTime.month,
            "day": currentDateTime.day,
            "hour": hour,
            "minute": 0,
            "second": 0
        }
    return timestamp


def mlTrain():
    generator()

    preprocessResults = []
    trainResults = []

    expectedFeedQuantities = []
    nextFeeding = []

    for i in cowData:
        expectedFeedQuantities.append(randint(20, 30))
        nextFeeding.append(generateTimeStamp(i))

    return [preprocessResults, trainResults, nextFeeding, expectedFeedQuantities]


def generateCowBasedData():
    biases = mlTrain()

    cowExpects = []

    for i in cowData:
        cowExpects.append([biases[2][int(i)], biases[3][int(i)]])

    return cowExpects


def bodyGeneratorAllData():
    templateString = """<tr><td><a href="{cowID}">{name}</a></td><td>{nextFeeding}</td><td>{expectedFeedingQuantity}</td><td>{weight}</td><td>{age}</td></tr>"""

    cowExpects = generateCowBasedData()

    returnString = ""

    for x in cowData:
        returnString += templateString.format(
            cowID=x,
            name=cowData[x]["ID"],
            nextFeeding=cowExpects[int(x)][0],
            expectedFeedingQuantity=cowExpects[int(x)][1],
            weight=cowData[x]["Weight"],
            age=cowData[x]["Age"]
        )

    return returnString


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/AllData.html")
def allData():

    table = Markup("""<table style="background-color: white">
      <thead>
        <tr style="color: white; background-color: green">
          <td>ID</td>
          <td>Next Feeding</td>
          <td>Expected Feed Quantity</td>
          <td>Weight</td>
          <td>Age</td>
        </tr>
      </thead>
      <tbody>
        {tableBody}
      </tbody>
    </table>""".format(tableBody=bodyGeneratorAllData()))
    return render_template('AllData.html', tables=table)


@app.route("/<cowID>")
def cID(cowID):
    return render_template("CowID.html", cowID=cowID, percentage=randint(1, 7))


def generateTableody(cowID):
    pass


@app.route("/a/<cowID>")
def cIDAD(cowID):
    table = Markup("""<table style = "background-color: white;">
			<h2>All Data</h2>
			<thead>
				<tr style = "color: white; background-color: green;">
					<td>Feeding Times</td>
					<td>Feed Quantity</td>
					<td>Weight</td>
					<td>Age</td>
				</tr>
			</thead>
			<tbody>
				<tr>
					<td>{fT0}</td>
					<td>{fQ0}</td>
					<td>{w0}</td>
					<td>{a1}</td>
				</tr>
				<tr>
					<td>{fT1}</td>
					<td>{fQ1}</td>
					<td>{w1}</td>
					<td>{a1}</td>
				</tr>
				<tr>
					<td>{fT2}</td>
					<td>{fQ2}</td>
					<td>{w2}</td>
					<td>{a2}</td>
				</tr>
			</tbody>
		</table>""".format(
            fT0=cowData[cowID]["Timestamp"]['0'],
            fQ0=cowData[cowID]["Ammount Eaten"]['0'],
            w0=cowData[cowID]["Weight"],
            a0=cowData[cowID]["Age"],
            fT1=cowData[cowID]["Timestamp"]['1'],
            fQ1=cowData[cowID]["Ammount Eaten"]['1'],
            w1=cowData[cowID]["Weight"],
            a1=cowData[cowID]["Age"],
            fT2=cowData[cowID]["Timestamp"]['2'],
            fQ2=cowData[cowID]["Ammount Eaten"]['2'],
            w2=cowData[cowID]["Weight"],
            a2=cowData[cowID]["Age"],
        ))
    return render_template("CowIDData.html", tables=table, cowID=int(cowID)+1)


with open('cowData.json') as f:
    cowData = jsonLoad(f)

if __name__ == "__main__":
    # preprocessTempFile = pathJoin(mainTempDir, 'keras')
    # trainerTempFile = pathJoin(mainTempDir, 'estimator')

    # transformer(train, test, preprocessTempFile)
    # preprocessResults = preprocessTrainer(preprocessTempFile)
    # pprint.pprint(preprocessResults)

    # transformer(train, test, trainerTempFile)
    # trainResults = trainerTrainer(trainerTempFile)
    # pprint.pprint(trainResults)

    app.run(port=80)

rmtree(mainTempDir)
