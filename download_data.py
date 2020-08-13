import sys
import logging
from collections import namedtuple
from itertools import chain
from os import makedirs

from networkx import MultiDiGraph, DiGraph
from networkx.algorithms import shortest_path

from research.knowledge_base import SparqlEndpoint
from research.rl_memory import SparqlKB


from record_store import DATE_DECADE, NAME_FIRST_LETTER

Experiment = namedtuple('Experiment', 'name, edges, start_vars, end_var, augment')
SparqlGraph = namedtuple('SparqlGraph', 'name, graph, start_vars, end_var, augment, actions')
Action = namedtuple('Action', 'type, subject, property, object, result')

LOGGER = logging.getLogger(__name__)

KB_SOURCE = SparqlEndpoint('http://162.233.132.179:8890/sparql')
KB_ADAPTOR = SparqlKB(KB_SOURCE)

NAME_PROP = '<http://xmlns.com/foaf/0.1/name>'
RELEASE_DATE_PROP = '<http://dbpedia.org/ontology/releaseDate>'
ALBUM_PROP = '<http://dbpedia.org/ontology/album>'
ARTIST_PROP = '<http://dbpedia.org/ontology/artist>'
HOMETOWN_PROP = '<http://dbpedia.org/ontology/hometown>'
COUNTRY_PROP = '<http://dbpedia.org/ontology/country>'
TYPE_PROP = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'


def to_sparql_graph(name, edges, start_vars, end_var, augment=None):
    graph = MultiDiGraph()
    for subj, prop, obj in edges:
        graph.add_edge(subj, obj, prop=prop)
    return SparqlGraph(
        name,
        graph,
        start_vars,
        end_var,
        augment,
        graph_to_actions(graph, start_vars, end_var),
    )


def networkx_to_sparql(sparql_graph):
    lines = []
    result_vars = ' '.join(
        f'?{var}' for var
        in sparql_graph.start_vars + [sparql_graph.end_var]
    )
    lines.append(f'SELECT DISTINCT {result_vars} WHERE {{')
    variables = set()
    names = set()
    for src, dst, _ in sparql_graph.graph.edges:
        variables.add(src)
        variables.add(dst)
        for _, edge_data in sparql_graph.graph.get_edge_data(src, dst).items():
            if edge_data['prop'] == NAME_PROP:
                names.add(dst)
            lines.append(f'    ?{src} {edge_data["prop"]} ?{dst} .')
    for name in names:
        lines.append(f'    FILTER ( lang(?{name}) = "en" )')
    lines.append(f'}}')
    return '\n'.join(lines)


def is_ambiguous(name):
    query = f'''
        SELECT DISTINCT ?concept WHERE {{
            ?concept {NAME_PROP} {name}
        }} LIMIT 2
    '''
    return len(list(KB_SOURCE.query_sparql(query))) != 1


def get_edge_prop(graph, src, dst):
    return graph.get_edge_data(src, dst)[0]['prop']


def add_path_to_action_graph(graph, action_graph, path):
    i = 0
    while i < len(path) - 1:
        curr_node = path[i]
        next_node = path[i + 1]
        if i + 2 < len(path):
            next_next_node = path[i + 2]
        else:
            next_next_node = None
        should_query_child = (
            next_next_node is not None
            and graph.has_edge(curr_node, next_node)
            and graph.has_edge(next_next_node, next_node)
        )
        if next_node == path[-1]:
            # use child
            prop = get_edge_prop(graph, curr_node, next_node)
            action_graph.add_edge(
                curr_node, next_node,
                action=Action('use', curr_node, prop, next_node, next_node),
            )
            i += 1
        elif graph.has_edge(next_node, curr_node):
            # query self
            prop = get_edge_prop(graph, next_node, curr_node)
            action_graph.add_edge(
                curr_node, next_node,
                action=Action('query', '_environment', prop, curr_node, next_node),
            )
            i += 1
        elif should_query_child:
            # query child
            prop = get_edge_prop(graph, curr_node, next_node)
            action_graph.add_edge(
                curr_node, next_next_node,
                action=Action('query', curr_node, prop, next_node, next_next_node),
            )
            i += 2
        elif graph.has_edge(curr_node, next_node):
            # retrieve child
            prop = get_edge_prop(graph, curr_node, next_node)
            action_graph.add_edge(
                curr_node, next_node,
                action=Action('retrieve', curr_node, prop, next_node, next_node),
            )
            i += 1
        else:
            ValueError(f'not sure what to do at {curr_node}')


def graph_to_actions(graph, start_vars, end_var):
    action_graph = build_action_graph(graph, start_vars, end_var)
    return sequentialize_actions(start_vars, end_var, action_graph)


def build_action_graph(graph, start_vars, end_var):
    action_graph = DiGraph()
    for start_var in start_vars:
        path = shortest_path(graph.to_undirected(), start_var, end_var)
        add_path_to_action_graph(graph, action_graph, path)
    return action_graph


def sequentialize_actions(start_vars, end_var, action_graph):
    queue = list(chain(*(
        action_graph.successors(start_var)
        for start_var in start_vars
    )))
    visited = set(start_vars)
    actions = []
    while queue:
        curr_node = queue.pop(0)
        if curr_node in visited:
            continue
        while all(node in visited for node in action_graph.predecessors(curr_node)):
            visited.add(curr_node)
            actions.append([
                action_graph.get_edge_data(predecessor, curr_node)['action']
                for predecessor in action_graph.predecessors(curr_node)
            ])
            if curr_node == end_var:
                break
            successors = list(action_graph.successors(curr_node))
            curr_node = successors[0]
            queue.extend(successors[1:])
        if curr_node != end_var:
            queue.append(curr_node)
    return actions


def all_sparql_results(query_template):
    limit = 100
    offset = 0
    query = query_template + f' LIMIT {limit} OFFSET {offset}'
    while True:
        results = KB_SOURCE.query_sparql(query)
        try:
            next_value = next(iter(results))
        except StopIteration:
            return
        yield next_value
        yield from results
        offset += limit
        query = query_template + f' LIMIT {limit} OFFSET {offset}'


def download_qa_pairs(sparql_graph):
    properties = dict(
        (
            start_var,
            set(
                edge[2] for edge
                in sparql_graph.graph.in_edges(start_var, data='prop')
            ),
        )
        for start_var in sparql_graph.start_vars
    )
    for result in all_sparql_results(networkx_to_sparql(sparql_graph)):
        answer = result[sparql_graph.end_var].rdf_format
        question = set()
        for start_var in sparql_graph.start_vars:
            start_rdf = result[start_var].rdf_format
            for prop in properties[start_var]:
                question.add((prop, start_rdf))
        yield question, answer


def check_query(cache, actions):
    query_terms = {}
    for action in actions:
        if action.property not in cache[action.subject]:
            return None
        query_terms[action.property] = cache[action.subject][action.property]
    LOGGER.debug(f'querying for {query_terms}')
    return KB_ADAPTOR.query(query_terms)


def check_retrieve(cache, actions):
    assert len(actions) == 1
    action = next(iter(actions))
    if action.property not in cache[action.subject]:
        return None
    LOGGER.debug(f'retrieving {cache[action.subject][action.property]}')
    return KB_ADAPTOR.retrieve(cache[action.subject][action.property])


def get_answer(question, sparql_graph):
    cache = {
        '_environment': dict(question),
    }
    for step, step_actions in enumerate(sparql_graph.actions, start=1):
        LOGGER.debug(f'Step {step}')
        LOGGER.debug('\n'.join(f'    {action}' for action in step_actions))
        assert len(set(action.type for action in step_actions)) == 1
        assert len(set(action.result for action in step_actions)) == 1
        for action in step_actions:
            if action.subject not in cache:
                raise ValueError('need {action.subject} but not found in cache')
        action = step_actions[0]
        result_var = action.result
        if action.type == 'query':
            result = check_query(cache, step_actions)
        elif action.type == 'retrieve':
            result = check_retrieve(cache, step_actions)
        elif action.type == 'use':
            LOGGER.debug(f'applying augment, if any')
            if sparql_graph.augment is None:
                answer_prop = action.property,
                answer = cache[action.subject].get(action.property, None)
            elif not all(attr in result for attr in sparql_graph.augment.old_attrs):
                return None
            else:
                answer_prop, answer = sparql_graph.augment.transform(cache[action.subject])
            if answer is None:
                return None
            else:
                return answer_prop, answer
        else:
            raise ValueError(step_actions)
        LOGGER.debug(f'result: {result}')
        if result is None:
            return None
        cache[result_var] = result
    raise ValueError('ran out of actions before use')


def get_valid_data(sparql_graph):
    qas = {}
    total = 0
    valid = 0
    for question, _ in download_qa_pairs(sparql_graph):
        q_list = tuple(sorted(question))
        if q_list in qas:
            continue
        total += 1
        LOGGER.info(f'trying to answer {question}')
        ambiguous = any(
            prop == NAME_PROP and is_ambiguous(obj)
            for prop, obj in question
        )
        if ambiguous:
            qas[q_list] = None
            continue
        answer = get_answer(question, sparql_graph)
        if answer is not None:
            _, answer = answer
            yield question, answer
            valid += 1
        qas[q_list] = answer
        if valid % 100 == 0:
            LOGGER.info(f'processed {valid} valid albums out of {total} albums')


def download_data(sparql_graph):
    makedirs('data', exist_ok=True)
    with open('data/' + sparql_graph.name, 'w') as fd:
        fd.write('(\n')
        for qa_tuple in get_valid_data(sparql_graph):
            fd.write(f'    {repr(qa_tuple)},\n')
        fd.write(')\n')


def find_leaves(graph):
    leaves = []
    for node in graph.nodes:
        try:
            next(iter(graph.successors(node)))
        except StopIteration:
            leaves.append(node)
    return leaves


EXP_RELEASE_DATE = to_sparql_graph(
    'album_date',
    [
        ('track', ALBUM_PROP, 'album_uri'),
        ('album_uri', NAME_PROP, 'album_name'),
        ('album_uri', RELEASE_DATE_PROP, 'release_date'),
    ],
    ['album_name'],
    'release_date',
    DATE_DECADE,
)


EXP_ARTIST = to_sparql_graph(
    'album_artist',
    (
        ('track_uri', ALBUM_PROP, 'album_uri'),
        ('album_uri', NAME_PROP, 'album_name'),
        ('album_uri', ARTIST_PROP, 'artist_uri'),
        ('artist_uri', NAME_PROP, 'artist_name'),
    ),
    ['album_name'],
    'artist_name',
    NAME_FIRST_LETTER,
)

EXP_COUNTRY = to_sparql_graph(
    'album_country',
    (
        ('track_uri', ALBUM_PROP, 'album_uri'),
        ('album_uri', NAME_PROP, 'album_name'),
        ('album_uri', ARTIST_PROP, 'artist_uri'),
        ('artist_uri', HOMETOWN_PROP, 'hometown_uri'),
        ('hometown_uri', COUNTRY_PROP, 'country_uri'),
        ('country_uri', NAME_PROP, 'country_name'),
    ),
    ['album_name'],
    'country_name',
)


EXP_OTHER_ALBUM = to_sparql_graph(
    'album_date_album',
    (
        ('track', ALBUM_PROP, 'album_uri'),
        ('album_uri', NAME_PROP, 'album_name'),
        ('album_uri', ARTIST_PROP, 'artist_uri'),
        ('other_track', ALBUM_PROP, 'other_album_uri'),
        ('other_album_uri', NAME_PROP, 'other_album_name'),
        ('other_album_uri', RELEASE_DATE_PROP, 'other_release_date'),
        ('other_album_uri', ARTIST_PROP, 'artist_uri'),
    ),
    ['album_name', 'other_release_date'],
    'other_album_name',
    NAME_FIRST_LETTER,
)


def main():
    if sys.argv[1] == 'release-date':
        sparql_graph = EXP_RELEASE_DATE
    elif sys.argv[1] == 'artist':
        sparql_graph = EXP_ARTIST
    elif sys.argv[1] == 'country':
        sparql_graph = EXP_COUNTRY
    elif sys.argv[1] == 'other-album':
        sparql_graph = EXP_OTHER_ALBUM
    else:
        print(f'unknown dataset "{sys.argv[1]}"')
        print(f'possible values are: release-date, artist, country, other-album')
        exit(1)
    download_data(sparql_graph)


if __name__ == '__main__':
    main()
