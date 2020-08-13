import re
from ast import literal_eval
from pathlib import Path

from research import State, Action, Environment
from research import SparqlLTM
from research import RandomMixin


class RecordStore(Environment, RandomMixin):

    def __init__(self, data_file=None, num_albums=1000, whitelist=None, *args, **kwargs):
        # pylint: disable = keyword-arg-before-vararg
        super().__init__(*args, **kwargs)
        # parameters
        assert data_file is not None
        self.data_file = Path(data_file).resolve()
        self.num_albums = num_albums
        # database
        self.questions = []
        self.answers = {}
        self.actions = set()
        self.whitelist = whitelist
        # variables
        self.question = None
        self.location = None
        self.reset()

    def get_state(self):
        return self._cache_state(
            (self.question, self.location),
            (lambda: State(**dict((
                *self.question,
                ('<http://dbpedia.org/ontology/location>', self.location),
            )))),
        )

    def get_actions(self):
        if self.location == self.answers[self.question]:
            return []
        actions = []
        for action_str in self.actions:
            assert isinstance(action_str, str), action_str
            actions.append(Action(action_str))
        return actions

    def react(self, action):
        self.location = action.name
        return -10

    def reset(self):
        qnas = []
        with self.data_file.open(encoding='utf-8') as fd:
            for question, answer in literal_eval(fd.read()):
                question = tuple(sorted(question))
                qnas.append((question, answer))
        if self.whitelist:
            qnas = [(q, a) for q, a in qnas if q in self.whitelist]
        if self.num_albums is None:
            self.answers = {question: answer for question, answer in qnas}
        elif self.num_albums > len(qnas) // 2: # FIXME parameterizable
            self.answers = {
                question: answer for question, answer
                in self.rng.sample(qnas, len(qnas) // 2)
            }
        else:
            self.answers = {
                question: answer for question, answer
                in self.rng.sample(qnas, self.num_albums)
            }
        self.questions = sorted(self.answers.keys())
        self.actions = set(self.answers.values())

    def start_new_episode(self):
        self.question = self.rng.choice(self.questions)
        self.location = '"__start__"'

    def visualize(self):
        raise NotImplementedError()


def first_letter(attr_dict):
    type_prop = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
    name_prop = '<http://xmlns.com/foaf/0.1/name>'
    if name_prop not in attr_dict:
        return None
    if type_prop not in attr_dict:
        return None
    literal = attr_dict[name_prop]
    obj_type = attr_dict[type_prop][1:-1].rsplit('/', maxsplit=1)[-1]
    match = re.fullmatch('"[^a-z]*([a-z]).*"(([@^][^"]*)?)', literal, flags=re.IGNORECASE)
    if not match:
        return None
    prop = f'<http://xmlns.com/foaf/0.1/{obj_type}/firstLetter>'
    initial = match.group(1).upper()
    metadata = match.group(2)
    return prop, f'"{initial}"{metadata}'


def date_to_decade(attr_dict):
    release_date_prop = '<http://dbpedia.org/ontology/releaseDate>'
    if release_date_prop not in attr_dict:
        return None
    date = attr_dict[release_date_prop]
    if not re.fullmatch('"([0-9]{3}).*"(([@^][^"]*)?)', date):
        return None
    release_decade_prop = '<http://dbpedia.org/ontology/releaseDecade>'
    decade = re.sub('^"([0-9]{3}).*"(([@^][^"]*)?)$', r'"\g<1>0-01-01"\2', date)
    return release_decade_prop, decade


INTERNAL_ACTIONS = set([
    'copy',
    'delete',
    'retrieve',
    'next-retrieval',
    'prev-retrieval',
])


def feature_extractor(state, action=None):
    features = set()
    features.add('_bias')
    internal = action is None or action.name in INTERNAL_ACTIONS
    external = action is None or action.name not in INTERNAL_ACTIONS
    for attribute, value in state:
        features.add((attribute, value))
    value = 1 / len(features)
    return {feature: value for feature in features}

'''
def feature_extractor(state, action=None):
    features = set()
    features.add('_bias')
    for attribute, value in state:
        features.add(attribute)
        features.add((attribute, value))
    value = 1 / len(features)
    return {feature: value for feature in features}
'''


NAME_FIRST_LETTER = SparqlLTM.Augment(
    [
        '<http://xmlns.com/foaf/0.1/name>',
        '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>',
    ],
    first_letter,
)

DATE_DECADE = SparqlLTM.Augment(
    ['<http://dbpedia.org/ontology/releaseDate>'],
    date_to_decade,
)
