import os
import sys
import pandas as pd

from linate import IdeologicalEmbedding, AttitudinalEmbedding

opj = os.path.join

EXPS = ['ChileOwn', 'FranceOwn', 'GermanyOwn', 'ItalyOwn', 'Spain', 'UKOwn']

# experiment params
params_ide_model = {
    "n_latent_dimensions": 2,
    "in_degree_threshold": 3,
    "check_input": True,
    "random_state": None,
    "engine": 'auto',
    "out_degree_threshold": None,
    "force_bipartite": True,
    "standardize_mean": True,
    "standardize_std": False,
    "force_full_rank": False
}

params_att_model = {
    "N": 2,
}

# (0) set experiment and data
experiment = sys.argv[1]

if experiment == 'linate_tutorial':

    PARENT_FOLDER = "/home/jimena/data/linate_tutorial"

    path_to_reference_group = opj(PARENT_FOLDER, 'reference_group.csv')

    path_to_group_attitudes = opj(PARENT_FOLDER, 'group_attitudes.csv')
    group_attitudes_columns = {'k': 'entity'}

    params_graph = {
        "path_to_network_data": opj(PARENT_FOLDER, 'bipartite_graph.csv'),
        "network_file_header_names":  {'target': 'i', 'source': 'j'},
    }

elif experiment in EXPS:

    PARENT_FOLDER = "/home/jimena/data/mds_PLOSOne_ASONAM_2022_dataset_V2_jimena/Publishable"

    path_to_reference_group = opj(
        PARENT_FOLDER,
        f'{experiment}_reference_group.csv')

    path_to_group_attitudes = opj(
        PARENT_FOLDER,
        f'{experiment}_group_attitudes.csv')

    if experiment == "ChileOwn":
        group_attitudes_columns = {
            'k': 'entity',
            'marporLA2020_per403': 'issue_1',
            'marporLA2020_per502': 'issue_2',
        }
    else:
        group_attitudes_columns = {
            'k': 'entity',
            'marpor2020_welfare': 'issue_1',
            'marpor2020_intpeace': 'issue_2'
        }
    params_graph = {
        "path_to_network_data": opj(
            PARENT_FOLDER, f'Bi_graph_{experiment}.csv.zip'),
        "network_file_header_names":  {'target': 'i', 'source': 'j'},
    }
else:
    raise ValueError(f"Wrong experiment '{experiment}'")


######## IDE ############

# (1) load bipartite graph and fit ideological embedding
ide_model = IdeologicalEmbedding(**params_ide_model)
bipartite = ide_model.load_input_from_file(**params_graph)
# /!\ HOT FIX very ugly to fix later
# convert str references ids to strings without dots
bipartite = bipartite.assign(
    target=bipartite.target.astype(float).astype(int).astype(str))
ide_model.fit(bipartite)

# (2) get users and references coordinates in ideological embedding
ref_coords_ide = ide_model.ideological_embedding_target_latent_dimensions_
ref_coords_ide['entity'] = ref_coords_ide.index
users_coords_ide = ide_model.ideological_embedding_source_latent_dimensions_
users_coords_ide['entity'] = users_coords_ide.index

########## ATT ####################

# (3) load and format data

# load table of group's coordinates in attitudinal space
print(f"Loading group's attitudes from {path_to_group_attitudes}...")
group_attitudes = pd.read_csv(path_to_group_attitudes, dtype={"k": str})
group_attitudes.rename(columns=group_attitudes_columns, inplace=True)
group_attitudes = group_attitudes[['entity', 'issue_1', 'issue_2']]

# remove groups with no attitudinal information
g0 = group_attitudes.entity.nunique()
group_attitudes.dropna(inplace=True)
print(f"Drop {g0 - group_attitudes.entity.nunique()} \
groups with nan attitudinal coordinates.")
valid_grups = group_attitudes.entity.unique().tolist()
print(f"Valid grups: {valid_grups}")

# load table of reference's groups
print(f"Loading reference group from {path_to_reference_group}...")
ref_group = pd.read_csv(path_to_reference_group, dtype=str)
ref_group.rename(columns={'i': 'entity', 'k': 'group'}, inplace=True)

# remove references from groups without coordinates in attitudinal embedding
r0 = ref_group.entity.nunique()
ref_group = ref_group.query(f"group in {tuple(valid_grups)}")
valid_refs = ref_group.entity.unique()

# compute groups coordinates in ideological embedding
ref_coords_ide = ref_coords_ide.merge(ref_group, "left")
group_coords_ide = ref_coords_ide.groupby('group').mean()

# (4) fit mapping from ideological embedding to attitudinal embedding
att_model = AttitudinalEmbedding(**params_att_model)

X = att_model.convert_to_group_ideological_embedding(
    ref_coords_ide.rename(columns={'group': 'k'}),
    ref_group)
Y = group_attitudes

att_model.fit(X, Y)

# (5) get users and references coordinates in attitudinal embedding
ref_coords_att = att_model.transform(ref_coords_ide)
users_coords_att = att_model.transform(users_coords_ide)
