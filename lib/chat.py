import os
import sys

import tensorflow as tf

from lib import data_utils
from lib.seq2seq_model_utils import create_model, get_predicted_sentence


def sample_index(probabilies):
    from random import random
    total = sum(probabilies)

    index = 0
    value = random()

    while True:
        value -= probabilies[index] / total
        if value <= 0:
            break
        else:
            index += 1

    return index


class ChatService:

    def __init__(self, args, session):
        self.args = args
        self.args.batch_size = 1
        self.session = session
        self.model = create_model(session, self.args)

        vocab_path = os.path.join(
            args.data_dir,
            "vocab%d.in" % args.vocab_size,
        )

        self.vocab, self.rev_vocab = data_utils.initialize_vocabulary(
            vocab_path
        )

    def _get_predicted_sentence(self, sentence):
        return get_predicted_sentence(
            self.args, sentence, self.vocab,
            self.rev_vocab, self.model, self.session,
        )

    def _decode_output_to_text(self, decode_output):
        decode_output = decode_output.replace('_GO', '')
        decode_output = decode_output.replace(' _EOS', '')
        decode_output = decode_output.replace(' _PAD', '')

        return decode_output

    def get_response(self, sentence):
        predicted_sentence = self._get_predicted_sentence(sentence)

        if isinstance(predicted_sentence, list):
            selected_index = sample_index(
                [x['prob'] for x in predicted_sentence]
            )
            return self._decode_output_to_text(
                predicted_sentence[selected_index]['dec_inp']
            )

        return predicted_sentence


def chat(args):
    with tf.Session() as session:
        service = ChatService(args, session)

        while True:
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

            if not session:
                break

            print(service.get_response(sentence))


def self_chat(args):
    with tf.Session() as session:
        service = ChatService(args, session)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while True:
            sentence = service.get_response(sentence)
            print('>', sentence)
            if sys.stdin.readline() == 'q':
                break
