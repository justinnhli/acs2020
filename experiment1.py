#!/usr/bin/env python3

from ast import literal_eval
from datetime import datetime
from pathlib import Path

from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

from permspace import PermutationSpace
from clusterun import sequencerun

from research import SparqlEndpoint
from research import train_and_evaluate
from research import epsilon_greedy, TabularQLearningAgent, LinearQLearner
from research import MemoryArchitectureMetaEnvironment as memory_architecture, SparqlLTM

from record_store import RecordStore, feature_extractor
from record_store import DATE_DECADE, NAME_FIRST_LETTER


with open('data/intersection') as fd:
    WHITELIST = literal_eval(fd.read())


def determine_augment(params):
    if params.data_file == 'album_date':
        return [DATE_DECADE]
    elif params.data_file in ['album_artist', 'album_date_album']:
        return [NAME_FIRST_LETTER]
    else:
        return None


def create_agent(params):
    return epsilon_greedy(LinearQLearner)(
        # Epsilon Greedy
        exploration_rate=0.05,
        # Linear Q Learner
        learning_rate=0.1,
        discount_rate=0.9,
        feature_fn=feature_extractor,
        # Random Mixin
        random_seed=params.random_seed,
    )


def create_env(params):
    record_store = RecordStore(
        data_file='data/' + params.data_file,
        num_albums=params.num_albums,
        whitelist=WHITELIST,
        random_seed=params.random_seed,
    )
    return memory_architecture(
        record_store,
        ltm=SparqlLTM(
            SparqlEndpoint('http://localhost:8890/sparql'),
            augments=determine_augment(params),
        ),
        # memory architecture
        max_internal_actions=params.max_internal_actions,
        buf_ignore=['scratch'],
    )


def run_experiment(params):
    agent = create_agent(params)
    results_dir = Path(__file__).parent / 'results' / params.results_folder
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes = range(
        0,
        # end is num_episodes, plus however it goes over by
        params.num_episodes + params.eval_frequency // 2,
        params.eval_frequency,
    )
    trial_results = train_and_evaluate(
        create_env(params),
        agent,
        num_episodes=params.num_episodes,
        eval_frequency=params.eval_frequency,
        min_return=params.min_return,
    )
    output_file = results_dir.joinpath(params.uniqstr_ + '.csv')
    try:
        for episode, mean_return in zip(episodes, trial_results):
            with output_file.open('a') as fd:
                fd.write(f'{datetime.now().isoformat("_")} {episode} {mean_return}\n')
    except QueryBadFormed as err:
        print('ERROR')
        print(err)
        with output_file.open('a') as fd:
            fd.write(str(err))
            fd.write('\n')


PSPACE = PermutationSpace(
    ['random_seed', 'data_file',],
    random_seed=[
        0.35746869278354254, 0.7368915891545381, 0.03439267552305503, 0.21913569678035283, 0.0664623502695384,
        0.53305059438797, 0.7405341747379695, 0.29303361447547216, 0.014835598224628765, 0.5731489218909421,
        0.7636381976146833, 0.35714236561930957, 0.5160608307412042, 0.7820994131649518, 0.31390169902962717,
        0.5400876377274924, 0.6525757873962879, 0.19218707681741432, 0.8670148169024791, 0.1790981637428084,
        0.9134217950356655, 0.040659298111523356, 0.06483438648885109, 0.43867544728173746, 0.4648996620113045,
        0.12592474382215701, 0.75692510690223, 0.09073875189436231, 0.3888019332434871, 0.023769648152276224,
        0.875555147892463, 0.8366393362290254, 0.5286188504870308, 0.34338492322440306, 0.661316883315625,
        0.729196739896136, 0.2112397121528542, 0.22586909337188776, 0.9702411834858093, 0.7004826619335851,
        0.39823445434135263, 0.7599284542986776, 0.5200829278658589, 0.9263527832114413, 0.16836668813041167,
        0.37993543222011084, 0.05646030607329311, 0.8380140269416136, 0.06850735156933208, 0.8509431330734283,
        0.7412794617644994, 0.2581948390155667, 0.730942481453577, 0.22603438819536303, 0.03423539666033948,
        0.302059151008751, 0.355906014056683, 0.08587605919951402, 0.5117755667491667, 0.8872689255632645,
        0.2912805392817581, 0.4129551853107706, 0.48796957175363065, 0.4007943172230767, 0.8605908670991194,
        0.24670183964853332, 0.16422009968131168, 0.7822393190331338, 0.9934975000705282, 0.06825588105012037,
        0.21311293630928718, 0.9234705997701798, 0.8326358654854799, 0.9071456994646435, 0.16481506944276747,
        0.8094178195801208, 0.5599773672976621, 0.411978414613525, 0.8196096292357119, 0.7986699933718194,
        0.8028611631686207, 0.4945949995685762, 0.22196103928134492, 0.645337288567758, 0.6435668607690285,
        0.5490678941921603, 0.7304438695693786, 0.2603483323175092, 0.7318268751726856, 0.12479832683538916,
    ],
    num_episodes=150000,
    eval_frequency=100,
    agent_type='kb',
    num_albums=100,
    max_internal_actions=(
        lambda data_file: {
            'album_date': 1,
            'album_artist': 2,
            'album_country': 4,
        }[data_file]
    ),
    data_file=[
        'album_date',
        'album_artist',
        'album_country',
    ],
    results_folder='experiment1',
    min_return=-100,
)


def importable_print(params):
    print(params)


def main():
    #sequencerun(importable_print, 'PSPACE')
    sequencerun(run_experiment, 'PSPACE')


if __name__ == '__main__':
    main()
