# encoding=utf8
import codecs
import pickle
import itertools
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences,load_sentences_test, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset,prepare_dataset_ner
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_skill,test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager
import os
import re
import codecs
from sklearn.model_selection import RepeatedKFold
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
flags = tf.flags
flags.DEFINE_boolean("clean",           True,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iob",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    50,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("map_ner_file", "maps_skill.pkl", "file for maps_ner")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("target_data", "All_skill.train"),  "Path for train data")
flags.DEFINE_string("train_file_loc",   os.path.join("source_data/loc", "loc.train"),  "Path for train data")
flags.DEFINE_string("dev_file_loc",     os.path.join("source_data/loc", "loc.dev"),    "Path for dev data")
flags.DEFINE_string("test_file_loc",    os.path.join("source_data/loc", "loc.test"),   "Path for test data")
flags.DEFINE_string("train_file_per",   os.path.join("source_data/per", "per.train"),  "Path for train data")
flags.DEFINE_string("dev_file_per",     os.path.join("source_data/per", "per.dev"),    "Path for dev data")
flags.DEFINE_string("test_file_per",    os.path.join("source_data/per", "per.test"),   "Path for test data")
flags.DEFINE_string("train_file_org",   os.path.join("source_data/org", "org.train"),  "Path for train data")
flags.DEFINE_string("dev_file_org",     os.path.join("source_data/org", "org.dev"),    "Path for dev data")
flags.DEFINE_string("test_file_org",    os.path.join("source_data/org", "org.test"),   "Path for test data")
FLAGS = tf.flags.FLAGS
#FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id,char_to_id_loc, tag_to_id_loc,char_to_id_per, tag_to_id_per,char_to_id_org, tag_to_id_org):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["num_chars_ner"] = len(char_to_id_loc)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["num_tags_ner"] = len(tag_to_id_loc)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag,precision_loc,precision_per,precision_org, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag,precision_loc,precision_per,precision_org)
    eval_lines ,results= test_skill(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1,results
def evaluate_ner(sess, model, name, data, id_to_tag_ner, logger):
    logger.info("evaluate_ner:{}".format(name))
    ner_results_ner = model.evaluate_ner(sess, data, id_to_tag_ner)
    eval_lines = test_ner(ner_results_ner, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev_ner":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev_ner f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test_ner":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test_ner f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train(X_train,X_dev,X_test):
    # load data sets
    train_sentences = X_train
    dev_sentences = X_dev
    test_sentences = X_test

    train_sentences_loc = load_sentences(FLAGS.train_file_loc, FLAGS.lower, FLAGS.zeros)
    dev_sentences_loc = load_sentences(FLAGS.dev_file_loc, FLAGS.lower, FLAGS.zeros)
    test_sentences_loc = load_sentences(FLAGS.test_file_loc, FLAGS.lower, FLAGS.zeros)
    train_sentences_org = load_sentences(FLAGS.train_file_org, FLAGS.lower, FLAGS.zeros)
    dev_sentences_org = load_sentences(FLAGS.dev_file_org, FLAGS.lower, FLAGS.zeros)
    test_sentences_org = load_sentences(FLAGS.test_file_org, FLAGS.lower, FLAGS.zeros)
    train_sentences_per = load_sentences(FLAGS.train_file_per, FLAGS.lower, FLAGS.zeros)
    dev_sentences_per = load_sentences(FLAGS.dev_file_per, FLAGS.lower, FLAGS.zeros)
    test_sentences_per = load_sentences(FLAGS.test_file_per, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)

    update_tag_scheme(train_sentences_loc, FLAGS.tag_schema)
    update_tag_scheme(test_sentences_loc, FLAGS.tag_schema)
    update_tag_scheme(train_sentences_per, FLAGS.tag_schema)
    update_tag_scheme(test_sentences_per, FLAGS.tag_schema)
    update_tag_scheme(train_sentences_org, FLAGS.tag_schema)
    update_tag_scheme(test_sentences_org, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
            dico_chars_train_loc = char_mapping(train_sentences_loc, FLAGS.lower)[0]
            dico_chars_loc, char_to_id_loc, id_to_char_loc = augment_with_pretrained(
                dico_chars_train_loc.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences_loc])
                )
            )
            dico_chars_train_per = char_mapping(train_sentences_per, FLAGS.lower)[0]
            dico_chars_per, char_to_id_per, id_to_char_per = augment_with_pretrained(
                dico_chars_train_per.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences_per])
                )
            )
            dico_chars_train_org = char_mapping(train_sentences_org, FLAGS.lower)[0]
            dico_chars_org, char_to_id_org, id_to_char_org = augment_with_pretrained(
                dico_chars_train_org.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences_org])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)
            _c_loc, char_to_id_loc, id_to_char_loc = char_mapping(train_sentences_loc, FLAGS.lower)
            _c_per, char_to_id_per, id_to_char_per = char_mapping(train_sentences_per, FLAGS.lower)
            _c_org, char_to_id_org, id_to_char_org = char_mapping(train_sentences_org, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        _t_loc, tag_to_id_loc, id_to_tag_loc = tag_mapping(train_sentences_loc)
        _t_per, tag_to_id_per, id_to_tag_per = tag_mapping(train_sentences_per)
        _t_org, tag_to_id_org, id_to_tag_org = tag_mapping(train_sentences_org)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag,char_to_id_loc, id_to_char_loc, tag_to_id_loc, id_to_tag_loc,char_to_id_per, id_to_char_per, tag_to_id_per, id_to_tag_per,char_to_id_org, id_to_char_org, tag_to_id_org, id_to_tag_org], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag,char_to_id_loc, id_to_char_loc, tag_to_id_loc, id_to_tag_loc,char_to_id_per, id_to_char_per, tag_to_id_per, id_to_tag_per,char_to_id_org, id_to_char_org, tag_to_id_org, id_to_tag_org = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data),len(dev_data), len(test_data)))
    train_data_loc = prepare_dataset_ner(
        train_sentences_loc, char_to_id_loc, tag_to_id_loc, FLAGS.lower
    )
    dev_data_loc = prepare_dataset_ner(
        dev_sentences_loc, char_to_id_loc, tag_to_id_loc, FLAGS.lower
    )
    test_data_loc = prepare_dataset_ner(
        test_sentences_loc, char_to_id_loc, tag_to_id_loc, FLAGS.lower
    )
    print("%i / %i / %i sentences_loc in train / dev / test." % (
        len(train_data_loc), len(dev_data_loc), len(test_data_loc)))
    train_data_per = prepare_dataset_ner(
        train_sentences_per, char_to_id_per, tag_to_id_per, FLAGS.lower
    )
    dev_data_per = prepare_dataset_ner(
        dev_sentences_per, char_to_id_per, tag_to_id_per, FLAGS.lower
    )
    test_data_per = prepare_dataset_ner(
        test_sentences_per, char_to_id_per, tag_to_id_per, FLAGS.lower
    )
    print("%i / %i / %i sentences_per in train / dev / test." % (
        len(train_data_per), len(dev_data_per), len(test_data_per)))
    train_data_org = prepare_dataset_ner(
        train_sentences_org, char_to_id_org, tag_to_id_org, FLAGS.lower
    )
    dev_data_org = prepare_dataset_ner(
        dev_sentences_org, char_to_id_org, tag_to_id_org, FLAGS.lower
    )
    test_data_org = prepare_dataset_ner(
        test_sentences_org, char_to_id_org, tag_to_id_org, FLAGS.lower
    )
    print("%i / %i / %i sentences_org in train / dev / test." % (
        len(train_data_org), len(dev_data_org), len(test_data_org)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    train_manager_loc = BatchManager(train_data_loc, FLAGS.batch_size)
    train_manager_per = BatchManager(train_data_per, FLAGS.batch_size)
    train_manager_org = BatchManager(train_data_org, FLAGS.batch_size)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id,char_to_id_loc, tag_to_id_loc,char_to_id_per, tag_to_id_per,char_to_id_org, tag_to_id_org)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    steps_per_epoch_loc = train_manager_loc.len_data
    steps_per_epoch_per = train_manager_per.len_data
    steps_per_epoch_org = train_manager_org.len_data
    model = create_model(Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, id_to_char_loc, id_to_char_per, id_to_char_org, logger)

    with tf.Session(config=tf_config, graph = model.graph ) as sess:

        sess.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            emb_weights = sess.run(model.char_lookup.read_value())
            emb_weights_ner = sess.run(model.char_lookup.read_value())
            emb_weights, emb_weights_ner = load_word2vec(config["emb_file"], id_to_char, id_to_char_loc,id_to_char_per,id_to_char_org, config["char_dim"],
                                                    emb_weights, emb_weights_ner)
            sess.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
        logger.info("start training")
        loss = []
        loss_loc = []
        loss_per = []
        loss_org = []
        for i in range(100):
            for batch_loc in train_manager_loc.iter_batch(shuffle=True):
                    step_loc, batch_loss_loc = model.run_step_ner(sess, True, batch_loc)
                    loss_loc.append(batch_loss_loc)
                    if step_loc % FLAGS.steps_check == 0:
                        iteration_loc = step_loc // steps_per_epoch_loc + 1
                        logger.info("iteration:{} step_loc:{}/{}, "
                                    "NER loss:{:>9.6f}".format(
                            iteration_loc, step_loc % steps_per_epoch_loc, steps_per_epoch_loc, np.mean(loss_loc)))
                        loss_loc = []
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration_1 = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                            "SKILL loss:{:>9.6f}".format(
                        iteration_1, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                loss = []
            precision_loc_dev = model.precision(sess, dev_manager, id_to_tag)
            precision_loc_test = model.precision(sess, test_manager, id_to_tag)
            for batch_per in train_manager_per.iter_batch(shuffle=True):
                    step_per, batch_loss_per = model.run_step_ner(sess, True, batch_per)
                    loss_per.append(batch_loss_per)
                    if step_per % FLAGS.steps_check == 0:
                        iteration_per = step_per // steps_per_epoch_per + 1
                        logger.info("iteration:{} step_per:{}/{}, "
                                    "NER loss:{:>9.6f}".format(
                            iteration_per, step_per % steps_per_epoch_per, steps_per_epoch_per, np.mean(loss_per)))
                        loss_per = []
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration_2 = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                            "SKILL loss:{:>9.6f}".format(
                        iteration_2, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                loss = []
            precision_per_dev = model.precision(sess, dev_manager, id_to_tag)
            precision_per_test = model.precision(sess, test_manager, id_to_tag)
            for batch_org in train_manager_org.iter_batch(shuffle=True):
                    step_org, batch_loss_org = model.run_step_ner(sess, True, batch_org)
                    loss_org.append(batch_loss_org)
                    if step_org % FLAGS.steps_check == 0:
                        iteration_org = step_org // steps_per_epoch_org + 1
                        logger.info("iteration:{} step_org:{}/{}, "
                                    "NER loss:{:>9.6f}".format(
                            iteration_org, step_org % steps_per_epoch_org, steps_per_epoch_org, np.mean(loss_org)))
                        loss_org = []
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration_3 = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                            "SKILL loss:{:>9.6f}".format(
                        iteration_3, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                loss = []
            precision_org_dev = model.precision(sess, dev_manager, id_to_tag)
            precision_org_test = model.precision(sess, test_manager, id_to_tag)
            best = evaluate(sess, model, "dev", dev_manager, id_to_tag,precision_loc_dev,precision_per_dev,precision_org_dev, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
                best_test,results= evaluate(sess, model, "test", test_manager, id_to_tag,precision_loc_test,precision_per_test,precision_org_test, logger)
                with open("CDTL_PSE-result.csv", "a",encoding='utf-8')as st_re:
                    st_re.write(str(results).replace("[", "").replace("]", ""))
                    st_re.write("\n")

def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            line = input("请输入测试句子:")
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            print('result:',result)
            break

def main():
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train(X_train,X_dev,X_test)


if __name__ == "__main__":
    sentences = load_sentences("target_data/All_skill.train", True, False)
    kf = RepeatedKFold(n_splits=5, n_repeats=1)
    for train_index, test_index in kf.split(sentences):
        X_trainall, X_test = np.array(sentences)[train_index], np.array(sentences)[test_index]
        kt = RepeatedKFold(n_splits=10, n_repeats=1)
        for train_index1, test_index1 in kt.split(X_trainall):
            X_train, X_dev = np.array(X_trainall)[train_index1], np.array(X_trainall)[test_index1]
            print(' X_train,X_test,X_dev', len(X_train), len(X_test), len(X_dev))
            main()









