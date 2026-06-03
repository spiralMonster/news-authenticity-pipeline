import csv
import tensorflow as tf


def _byte_features(value):
    byte_features=tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return byte_features


def ConvertDataToTFRecords(csv_data_path:str,tfrecord_path:str):
    tfrecord_writer=tf.io.TFRecordWriter(tfrecord_path)

    with open(csv_data_path) as csv_file:
        csv_reader=csv.DictReader(csv_file,delimiter=",",quotechar='"')

        for row in csv_reader:
            example=tf.train.Example(
                features=tf.train.Features(feature={
                    "text":_byte_features(row['text'].encode('utf-8')),
                    "label":_byte_features(row['label'].encode('utf-8'))
                })
            )

            tfrecord_writer.write(example.SerializeToString())

    tfrecord_writer.close()

    print(f"[INFO] Data converted into TFRecord...")

    return
